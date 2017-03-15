extern crate bit_set;
extern crate petgraph;
extern crate rand;
extern crate rayon;
#[cfg(test)]
#[macro_use]
extern crate quickcheck;

use bit_set::BitSet;
use petgraph::prelude::*;
use rand::{Rng, thread_rng, sample as rng_sample};
use rand::distributions::{Range, Sample, IndependentSample};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::iter::FromIterator;
use rayon::prelude::*;

pub trait TriggeringModel<N, E> {
    fn new<V: FromIterator<NodeIndex>>(g: &Graph<N, E>, source: NodeIndex) -> V;

    fn new_uniform<V: FromIterator<NodeIndex>>(g: &Graph<N, E>) -> V {
        let mut rng = thread_rng();
        let source = *rng.choose(&g.node_indices().collect::<Vec<_>>()).unwrap();
        Self::new(g, source)
    }
}

/// Generate a `k`-element sample from a graph `g` under model `M`
pub fn sample<N, E, M>(g: &Graph<N, E>,
                       k: usize)
                       -> (BTreeMap<NodeIndex, BTreeSet<NodeIndex>>, BTreeMap<NodeIndex, usize>)
    where N: Sync,
          E: Sync,
          M: TriggeringModel<N, E>
{
    let mut rng = thread_rng();
    let ind = g.node_indices().collect::<Vec<NodeIndex>>();
    let roots = rng_sample(&mut rng, &ind, k);

    let mut sets: Vec<BTreeSet<NodeIndex>> = Vec::with_capacity(k);
    roots.par_iter().map(|&&root| M::new(&g, root)).collect_into(&mut sets);

    let mut map = BTreeMap::new();
    let mut counts = BTreeMap::new();
    for (&root, set) in roots.into_iter().zip(sets) {
        for el in set {
            map.entry(el).or_insert_with(|| BTreeSet::new()).insert(root);
            *counts.entry(el).or_insert(0) += 1;
        }
    }

    (map, counts)
}

/// A RIS sampler following the IC model.
///
/// # Examples
///
/// ```rust
/// # extern crate petgraph;
/// # extern crate ris;
/// # fn main() {
/// use petgraph::prelude::*;
/// use ris::{IC, TriggeringModel};
/// let g: Graph<(), f32> = Graph::from_edges(&[
///     (0, 1, 0.4), (0, 2, 0.3),
///     (1, 0, 0.2), (1, 2, 0.8),
///     (2, 3, 0.5), (3, 1, 0.2)
/// ]);
///
/// let sample: Vec<NodeIndex> = IC::new(&g, NodeIndex::new(0));
/// # }
/// ```
#[derive(Clone)]
pub struct IC {}

impl<N, E: Copy + Into<f64>> TriggeringModel<N, E> for IC {
    fn new<V: FromIterator<NodeIndex>>(graph: &Graph<N, E>, source: NodeIndex) -> V {
        let mut activated = BitSet::new();
        activated.insert(source.index());
        let mut queue = VecDeque::from(vec![source]);
        let mut rng = thread_rng();

        let mut sample = Vec::new();

        while let Some(node) = queue.pop_front() {
            let mut uniform = Range::new(0.0, 1.0);
            // generate a list of new, unactivated neighbors
            let mut stack_ext: VecDeque<_> = graph.edges_directed(node, Incoming)
                .filter_map(|edge| {
                    if !activated.contains(edge.source().index()) &&
                       uniform.sample(&mut rng) <= (*edge.weight()).into() {
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

impl<N, E: Copy + Into<f64>> TriggeringModel<N, E> for LT {
    fn new<V: FromIterator<NodeIndex>>(graph: &Graph<N, E>, source: NodeIndex) -> V {
        let mut activated = BitSet::new();
        activated.insert(source.index());
        let mut rng = thread_rng();
        let uniform = Range::new(0.0, 1.0);

        let mut sample = Vec::new();
        let mut next = Some(source);
        while let Some(node) = next {
            // TODO: binary search instead of linear scan
            let goal = uniform.ind_sample(&mut rng);
            if goal <=
               graph.edges_directed(node, Incoming)
                .map(|edge| (*edge.weight()).into())
                .sum() {
                let mut activator = None;
                let mut sum = 0f64;
                for edge in graph.edges_directed(node, Incoming) {
                    sum += (*edge.weight()).into();
                    if sum >= goal {
                        if !activated.contains(edge.source().index()) {
                            activated.insert(edge.source().index());
                            activator = Some(edge.source());
                            break;
                        }
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

#[cfg(test)]
mod test {
    use super::*;
    use quickcheck::TestResult;
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
    }

    macro_rules! ris_props {
        ($m:ident, $M:ident) => {
            mod $m {
                use super::*;
                use petgraph::graph::node_index;
                use quickcheck::TestResult;
                use std::collections::BTreeSet;

                quickcheck! {
                    fn unique(g: Graph<(), f32>, source: usize) -> TestResult {
                        if source >= g.node_count() || g.edge_count() == 0 {
                            return TestResult::discard();
                        }
                        let g = reweight(g);

                        let source = node_index(source);
                        let sample: Vec<_> = $M::new(&g, source);
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
                        let sample: Vec<_> = $M::new(&g, source);

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

    ris_props!(ic, IC);
    ris_props!(lt, LT);
}
