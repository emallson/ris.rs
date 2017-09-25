extern crate bit_set;
extern crate petgraph;
extern crate rand;
extern crate rayon;
#[cfg(test)]
#[macro_use]
extern crate quickcheck;
extern crate vec_graph;
extern crate as_num;
#[cfg(feature = "hash")]
extern crate fnv;

#[cfg(not(feature = "hash"))]
use bit_set::BitSet;
#[cfg(feature = "hash")]
use fnv::FnvHashSet;
use petgraph::prelude::*;
use petgraph::graph::GraphIndex;
use petgraph::visit::{GraphRef, IntoNodeIdentifiers, Data, NodeCount};
use vec_graph::IntoNeighborEdgesDirected;
use rand::{Rng, thread_rng};
use rand::distributions::{Range, Sample, IndependentSample};
use rand::distributions::range::SampleRange;
use std::collections::VecDeque;
use std::iter::FromIterator;
use rayon::prelude::*;
use as_num::{AsNumInternal, AsNum};
use std::cell::RefCell;

pub trait TriggeringModel<G: GraphRef + Data<NodeWeight = N, EdgeWeight = E>, N, E>
    
    where G::NodeId: GraphIndex,
          G::EdgeId: GraphIndex
{
    fn new<V: FromIterator<G::NodeId>, R: Rng>(rng: &mut R, g: G, source: G::NodeId) -> V;
}

pub trait TriggeringModelUniform<G: GraphRef + Data<NodeWeight = N, EdgeWeight = E>,
                                 N,
                                 E,
                                 Ix = petgraph::graph::DefaultIx>
    : TriggeringModel<G, N, E>
    where G::NodeId: GraphIndex + From<Ix>,
          G::EdgeId: GraphIndex,
          Ix: SampleRange + PartialOrd + AsNumInternal<usize> + std::fmt::Debug,
          usize: AsNumInternal<Ix>
{
    fn new_uniform<V: FromIterator<G::NodeId>>(g: G) -> V
        where G::NodeId: From<Ix>,
              G: NodeCount
    {
        let mut rng = thread_rng();
        Self::new_uniform_with(&mut rng, g)
    }

    fn new_uniform_with<V: FromIterator<G::NodeId>, R: Rng>(rng: &mut R, g: G) -> V
        where G::NodeId: From<Ix>,
              G: NodeCount
    {
        let range = Range::<Ix>::new(0usize.as_num(), g.node_count().as_num());
        let source = range.ind_sample(rng).into();
        Self::new(rng, g, source)
    }
}

/// Generate a `k`-element sample from a graph `g` under model `M`
pub fn sample<G: GraphRef + Data<NodeWeight = N, EdgeWeight = E> + NodeCount + IntoNodeIdentifiers + Sync,
              N,
              E,
              M,
              U: FromIterator<G::NodeId> + Send>
    (g: G,
     k: usize)
     -> Vec<U>
    where N: Sync,
          E: Sync,
          M: TriggeringModel<G, N, E>,
          G::NodeId: GraphIndex + Send + Sync,
          G::EdgeId: GraphIndex
{
    let mut rng = thread_rng();
    let mut roots = Vec::with_capacity(k);
    let indices = g.node_identifiers().collect::<Vec<_>>();
    for _ in 0..k {
        roots.push(*rng.choose(&indices).unwrap());
    }

    let mut sets: Vec<U> = Vec::with_capacity(k);
    roots.par_iter().map(|&root| M::new(&mut thread_rng(), g, root)).collect_into(&mut sets);
    sets
}

/// A RIS sampler following the IC model.
///
/// # Examples
///
/// ```rust
/// # extern crate petgraph;
/// # extern crate rand;
/// # extern crate ris;
/// # fn main() {
/// use petgraph::prelude::*;
/// use rand::thread_rng;
/// use ris::{IC, TriggeringModel};
/// let g: Graph<(), f32> = Graph::from_edges(&[
///     (0, 1, 0.4), (0, 2, 0.3),
///     (1, 0, 0.2), (1, 2, 0.8),
///     (2, 3, 0.5), (3, 1, 0.2)
/// ]);
///
/// let sample: Vec<NodeIndex> = IC::new(&mut thread_rng(), &g, NodeIndex::new(0));
/// # }
/// ```
#[derive(Clone)]
pub struct IC {}

thread_local!(static MARK: RefCell<Vec<bool>> = RefCell::new(Vec::new()));
thread_local!(static VISIT: RefCell<Vec<usize>> = RefCell::new(Vec::new()));

impl<'a> TriggeringModel<&'a vec_graph::VecGraph<(), f32>, (), f32> for IC {
    fn new<V: FromIterator<vec_graph::NodeIndex>, R: Rng>(rng: &mut R,
                                                          graph: &'a vec_graph::VecGraph<(), f32>,
                                                          source: vec_graph::NodeIndex)
                                                          -> V {
        VISIT.with(|v| {
            MARK.with(|m| {
                let mut mark = m.borrow_mut();
                let mut visit = v.borrow_mut();
                // mark.reserve_len(graph.node_count());
                mark.resize(graph.node_count(), false);
                visit.resize(graph.node_count(), ::std::usize::MAX);

                let mut cur_pos = 0;
                let mut num_marked = 1;
                mark[source.index()] = true;
                visit[0] = source.index();
                let uniform = Range::new(0.0, 1.0);
                while cur_pos < num_marked {
                    let cur = visit[cur_pos];
                    cur_pos += 1;
                    let ref edges = graph.incoming_edges[cur];
                    let ref weights = graph.incoming_weights[cur];
                    for i in 0..edges.len() {
                        if uniform.ind_sample(rng) < weights[i] && !mark[edges[i]] {
                            mark[edges[i]] = true;
                            visit[num_marked] = edges[i];
                            num_marked += 1;
                        }
                    }
                }
                for i in 0..num_marked {
                    mark[visit[i]] = false;
                }

                V::from_iter(visit.iter().take(num_marked).map(|&u| vec_graph::NodeIndex::new(u)))
            })
        })
    }
}

impl<'a> TriggeringModelUniform<&'a vec_graph::VecGraph<(), f32>, (), f32, usize> for IC {}

impl<'a, N, E: Copy + Into<f64>> TriggeringModel<&'a petgraph::Graph<N, E>, N, E> for IC {
    fn new<V: FromIterator<petgraph::graph::NodeIndex>, R: Rng>(rng: &mut R,
                                                                graph: &'a petgraph::Graph<N, E>,
                                                                source: petgraph::graph::NodeIndex)
                                                                -> V {
        #[cfg(not(feature = "hash"))]
        let mut activated = BitSet::new();
        #[cfg(feature = "hash")]
        let mut activated = FnvHashSet::default();
        activated.insert(source.index());
        let mut queue = VecDeque::from(vec![source]);

        let mut sample = Vec::new();

        while let Some(node) = queue.pop_front() {
            let mut uniform = Range::new(0.0, 1.0);

            // generate a list of new, unactivated neighbors
            let mut stack_ext: VecDeque<_> = graph.edges_directed(node, Incoming)
                .filter_map(|edge| {
                    #[cfg(not(feature = "hash"))]
                    let contained = activated.contains(edge.source().index());
                    #[cfg(feature = "hash")]
                    let contained = activated.contains(&edge.source().index());
                    if !contained && uniform.sample(rng) <= (*edge.weight()).into() {
                        activated.insert(edge.source().index());
                        Some(edge.source().clone())
                    } else {
                        None
                    }
                })
                .collect();

            queue.append(&mut stack_ext);
            sample.push(node);
        }

        V::from_iter(sample.into_iter())
    }
}

/// LT Live-Edge model. It is assumed that the input graph has each in-neighborhood weighted s.t.
/// the sum of edge weights `W` satisfies `0 <= W <= 1`.
pub struct LT {}

/// Reweight the gaph to satisfy the LT assumptions. `alpha` is the probability of no edge being
/// selected.
pub fn reweight_lt<N: Clone, E: Into<f32> + Clone>(g: &Graph<N, E>, alpha: f32) -> Graph<N, f32> {
    let mut g: Graph<N, f32> = g.map(|_, n| n.clone(), |_, e| e.clone().into());
    for node in g.node_indices() {
        let sum: f32 = g.edges_directed(node, Incoming).map(|edge| *edge.weight()).sum();
        let mut edges = g.neighbors_directed(node, Incoming).detach();
        while let Some(edge) = edges.next_edge(&g) {
            g[edge] = g[edge] / sum * (1.0 - alpha);
        }
    }
    g
}

impl<G: GraphRef + NodeCount + IntoNeighborEdgesDirected + Data<NodeWeight=N, EdgeWeight=E>, N, E: Copy + Into<f64>> TriggeringModel<G, N, E> for LT
    where G::NodeId: GraphIndex,
          G::EdgeId: GraphIndex
{
    fn new<V: FromIterator<G::NodeId>, R: Rng>(rng: &mut R, graph: G, source: G::NodeId) -> V {
        #[cfg(not(feature = "hash"))]
        let mut activated = BitSet::new();
        #[cfg(feature = "hash")]
        let mut activated = FnvHashSet::default();
        activated.insert(source.index());
        let uniform = Range::new(0.0, 1.0);

        let mut sample = Vec::new();
        let mut next = Some(source);
        while let Some(node) = next {
// TODO: binary search instead of linear scan
            let goal = uniform.ind_sample(rng);
            if goal <=
               graph.edges_directed(node, Incoming)
                .map(|edge| (*edge.weight()).into())
                .sum() {
                let mut activator = None;
                let mut sum = 0f64;
                for edge in graph.edges_directed(node, Incoming) {
                    sum += (*edge.weight()).into();
                    if sum >= goal {
                        #[cfg(not(feature = "hash"))]
                        let contained = activated.contains(edge.source().index());
                        #[cfg(feature = "hash")]
                        let contained = activated.contains(&edge.source().index());
                        if !contained {
                            activated.insert(edge.source().index());
                            activator = Some(edge.source());
                        }
                        break;
                    }
                }

                next = activator;
            } else {
                next = None;
            }

            sample.push(node);
        }
        V::from_iter(sample.into_iter())
    }
}

macro_rules! impl_uniform {
    ($model:path, $num:ty) => {
        impl<G: GraphRef + NodeCount + IntoNeighborEdgesDirected + Data<NodeWeight=N, EdgeWeight=E>, N, E: Copy + Into<f64>> TriggeringModelUniform<G, N, E, $num> for $model 
            where G::NodeId: GraphIndex + From<$num>,
                  G::EdgeId: GraphIndex
                  {
                  }
    }
}

impl_uniform!(LT, usize);
impl_uniform!(LT, u16);
impl_uniform!(LT, u32);
impl_uniform!(LT, u64);
// impl_uniform!(IC, usize);
// impl_uniform!(IC, u16);
// impl_uniform!(IC, u32);
// impl_uniform!(IC, u64);

#[cfg(test)]
mod test {
    use super::*;
    use quickcheck::TestResult;
    use vec_graph::VecGraph;
    use std::cmp::Ordering;

    /// Re-weight a graph from the entire f32 range to just `[0, 1]`.
    pub fn reweight(g: Graph<(), f32>) -> Graph<(), f32> {
        let mut weights: Vec<_> = g.edge_references().map(|edge| edge.weight().abs()).collect();
        weights.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let max = *weights.last().unwrap();
        g.map(|_, &n| n, |_, &w| w.abs() / max)
    }

    quickcheck! {
        fn reweight_works(g: Graph<(), f32>) -> TestResult {
            if g.edge_count() == 0 {
                return TestResult::discard();
            }

            let g = reweight(g);

            TestResult::from_bool(!g.edge_references().any(|edge| edge.weight() < &0.0 || edge.weight() > &1.0))
        }

        fn accepts_vec_graph(g: Graph<(), f32>) -> TestResult {
            if g.edge_count() == 0 {
                return TestResult::discard();
            }
            let vg = VecGraph::from_petgraph(g);

            let _ic: Vec<_> = IC::new_uniform(&vg);
            let _lt: Vec<_> = LT::new_uniform(&vg);
            TestResult::passed()
        }
    }

    macro_rules! ris_props {
        ($m:ident, $M:ident) => {
            mod $m {
                use super::*;
                use petgraph::graph::node_index;
                use quickcheck::TestResult;
                use std::collections::BTreeSet;
                use rand::thread_rng;

                quickcheck! {
                    fn source_contained(g: Graph<(), f32>, source: usize) -> TestResult {
                        if source >= g.node_count() || g.edge_count() == 0 {
                            return TestResult::discard();
                        }
                        let g = reweight(g);

                        let source = node_index(source);
                        let sample: Vec<_> = $M::new(&mut thread_rng(), &g, source);
                        TestResult::from_bool(sample.contains(&source))
                    }
                    fn unique(g: Graph<(), f32>, source: usize) -> TestResult {
                        if source >= g.node_count() || g.edge_count() == 0 {
                            return TestResult::discard();
                        }
                        let g = reweight(g);

                        let source = node_index(source);
                        let sample: Vec<_> = $M::new(&mut thread_rng(), &g, source);
                        let set: BTreeSet<_> = sample.iter().cloned().collect();
                        if set.len() != sample.len()
                        {
                            println!("{:?}", sample);
                        }

                        TestResult::from_bool(set.len() == sample.len())
                    }

                    fn reverse_reachable(g: Graph<(), f32>, source: usize) -> TestResult {
                        if source >= g.node_count() || g.edge_count() == 0 {
                            return TestResult::discard();
                        }

                        let g = reweight(g);
                        let source = node_index(source);
                        let sample: Vec<_> = $M::new(&mut thread_rng(), &g, source);

    // due to the ordering of sampling, we know that each node has to be reachable by one
    // of the prior nodes (excluding, of course, the first node)
                        if sample[0] != source {
                            return TestResult::failed();
                        }

                        for (index, &node) in sample.iter().enumerate().skip(1) {
                            if !sample.iter().take(index).any(|&prior| g.contains_edge(node, prior)) {
                                return TestResult::failed();
                            }
                        }

                        TestResult::passed()
                    }
                }
            }
        }
    }

    // ris_props!(ic, IC);
    ris_props!(lt, LT);
}
