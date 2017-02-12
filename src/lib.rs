extern crate bit_set;
extern crate petgraph;
extern crate rand;
extern crate rayon;
#[cfg(test)]
#[macro_use]
extern crate quickcheck;

use bit_set::BitSet;
use petgraph::prelude::*;
use rand::{Rng, thread_rng};
use rand::distributions::{Range, Sample};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use rayon::prelude::*;

pub trait TriggeringModel<'a, N, E>: Iterator<Item = NodeIndex> + Sized {
    fn new(g: &'a Graph<N, E>, source: NodeIndex) -> Self;

    fn new_uniform(g: &'a Graph<N, E>) -> Self {
        let mut rng = thread_rng();
        let source = *rng.choose(&g.node_indices().collect::<Vec<_>>()).unwrap();
        Self::new(g, source)
    }
}

/// Generate a `k`-element sample from a graph `g` under model `M`
pub fn sample<'a, 'b, N, E, M>
    (g: &'a Graph<N, E>,
     k: usize)
     -> (BTreeMap<NodeIndex, BTreeSet<NodeIndex>>, BTreeMap<NodeIndex, usize>)
    where 'a: 'b,
          N: Sync,
          E: Sync,
          M: TriggeringModel<'b, N, E>
{
    let mut rng = thread_rng();
    let ind = g.node_indices().collect::<Vec<_>>();
    let roots = (0..k).map(|_| *rng.choose(&ind).unwrap()).collect::<Vec<_>>();

    let mut sets: Vec<BTreeSet<NodeIndex>> = Vec::with_capacity(k);
    roots.par_iter().map(|&root| M::new(&g, root).collect()).collect_into(&mut sets);

    let mut map = BTreeMap::new();
    let mut counts = BTreeMap::new();
    for (root, set) in roots.into_iter().zip(sets) {
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
/// let sample: Vec<NodeIndex> = IC::new(&g, NodeIndex::new(0)).collect();
/// # }
/// ```
#[derive(Clone)]
pub struct IC<'a, N: 'a, E: 'a + Into<f32> + Clone, R: Rng> {
    graph: &'a Graph<N, E>,
    queue: VecDeque<NodeIndex>,
    rng: R,
    // used to guarantee each edge is processed only once
    activated: BitSet,
}

impl<'a, N: 'a, E: 'a + Into<f32> + Clone> TriggeringModel<'a, N, E>
    for IC<'a, N, E, rand::ThreadRng> {
    fn new(g: &'a Graph<N, E>, source: NodeIndex) -> Self {
        IC::with_rng(g, source, rand::thread_rng())
    }
}

impl<'a, N, E: Into<f32> + Clone, R: Rng> IC<'a, N, E, R> {
    pub fn with_rng(g: &'a Graph<N, E>, source: NodeIndex, rng: R) -> Self {
        let mut act = BitSet::new();
        act.insert(source.index());
        IC {
            graph: g,
            queue: vec![source].into(),
            rng: rng,
            activated: act,
        }
    }
}

impl<'a, N, E: Into<f32> + Clone, R: Rng> Iterator for IC<'a, N, E, R> {
    type Item = NodeIndex;
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(node) = self.queue.pop_front() {
            let mut uniform = Range::new(0.0, 1.0);
            // generate a list of new, unactivated neighbors
            let mut stack_ext: VecDeque<_> = self.graph
                .edges_directed(node, Incoming)
                .filter_map(|edge| {
                    if !self.activated.contains(edge.source().index()) &&
                       uniform.sample(&mut self.rng) <= edge.weight().clone().into() {
                        self.activated.insert(edge.source().index());
                        Some(edge.source().clone())
                    } else {
                        None
                    }
                })
                .collect();

            self.queue.append(&mut stack_ext);

            Some(node)
        } else {
            None
        }
    }
}

pub struct LT<'a, N: 'a, E: 'a + Into<f32> + Clone, R: Rng> {
    graph: &'a Graph<N, E>,
    next: Option<NodeIndex>,
    rng: R,
    activated: BitSet,
}

impl<'a, N, E: Into<f32> + Clone> TriggeringModel<'a, N, E> for LT<'a, N, E, rand::ThreadRng> {
    fn new(g: &'a Graph<N, E>, source: NodeIndex) -> Self {
        LT::with_rng(g, source, rand::thread_rng())
    }
}

impl<'a, N, E: Into<f32> + Clone> LT<'a, N, E, rand::ThreadRng> {}

impl<'a, N, E: Into<f32> + Clone, R: Rng> LT<'a, N, E, R> {
    pub fn with_rng(g: &'a Graph<N, E>, source: NodeIndex, rng: R) -> Self {
        let mut act = BitSet::new();
        act.insert(source.index());
        LT {
            graph: g,
            next: Some(source),
            rng: rng,
            activated: act,
        }
    }
}

impl<'a, N, E: Into<f32> + Clone, R: Rng> Iterator for LT<'a, N, E, R> {
    type Item = NodeIndex;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(node) = self.next {
            // TODO: binary search instead of linear scan
            let goal: f32 = self.rng.gen();
            if goal <=
               self.graph
                .edges_directed(node, Incoming)
                .map(|edge| edge.weight().clone().into())
                .sum() {
                let mut activator = None;
                let mut sum = 0f32;
                for edge in self.graph.edges_directed(node, Incoming) {
                    sum += edge.weight().clone().into();
                    if sum >= goal {
                        if !self.activated.contains(edge.source().index()) {
                            self.activated.insert(edge.source().index());
                            activator = Some(edge.source().clone());
                            break;
                        }
                    }
                }

                self.next = activator;
            } else {
                self.next = None;
            }

            Some(node)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use petgraph::prelude::*;
    use petgraph::graph::node_index;
    use quickcheck::TestResult;
    use std::cmp::Ordering;

    use rand::isaac::IsaacRng;

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
                        let sample: Vec<_> = $M::new(&g, source).collect();
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
                        let sample: Vec<_> = $M::new(&g, source).collect();

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

    #[test]
    fn with_rng() {
        let rng = IsaacRng::new_unseeded();
        let g: Graph<(), f32> = Graph::from_edges(&[(1, 0, 0.8),
                                                    (2, 0, 0.3),
                                                    (0, 1, 0.2),
                                                    (2, 1, 0.7),
                                                    (3, 1, 0.95),
                                                    (1, 3, 0.2)]);
        let sample: Vec<_> = IC::with_rng(&g, node_index(0), rng).collect();

        // with the default seed (which is fixed), the sample is [0, 1, 3].
        assert!(sample == vec![node_index(0), node_index(1), node_index(3)]);
    }
}
