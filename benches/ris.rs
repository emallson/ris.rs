#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate bencher;
extern crate ris;
extern crate capngraph;
extern crate rand_mersenne_twister;
extern crate petgraph;
extern crate vec_graph;

use bencher::Bencher;
use ris::{IC, TriggeringModelUniform};
use std::collections::BTreeSet;
use rand_mersenne_twister::mersenne;
use vec_graph::VecGraph;

lazy_static! {
    static ref G: vec_graph::Graph<(), f32> = {
        VecGraph::oriented_from_edges(capngraph::load_edges("benches/data/soc-Slashdot0902.bin").unwrap(), petgraph::Direction::Incoming)
    };
}

fn slashdot_ic(bench: &mut Bencher) {
    let r = bench.iter(|| {
        let mut rng = mersenne();
        for _ in 0..1_000 {
            IC::new_uniform_with::<Vec<_>, _>(&mut rng, &*G);
        }
    });
    r
}

fn slashdot_ic_btree(bench: &mut Bencher) {
    // bench.iter(|| black_box((0..10_000).map(|_| IC::new_uniform::<Vec<_>>(&g))));
    let r = bench.iter(|| {
        let mut rng = mersenne();
        for _ in 0..1_000 {
            IC::new_uniform_with::<BTreeSet<_>, _>(&mut rng, &*G);
        }
    });
    r
}

benchmark_group!(slashdot, slashdot_ic, slashdot_ic_btree);
benchmark_main!(slashdot);
