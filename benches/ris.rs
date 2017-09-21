#[macro_use]
extern crate bencher;
extern crate ris;
extern crate quickcheck;
extern crate petgraph;
extern crate rand;

use quickcheck::{Arbitrary, StdGen};
use petgraph::prelude::*;
use rand::thread_rng;

use bencher::{black_box, Bencher};
use ris::*;

fn lt(bench: &mut Bencher) {
    let mut gen = StdGen::new(thread_rng(), 100);
    let mut g: Graph<(), f32> = Arbitrary::arbitrary(&mut gen);

    for _ in 0..100 {
        g = Arbitrary::arbitrary(&mut gen);
    }

    let g = reweight_lt(&g, 0.1);
    bench.iter(|| black_box(LT::new_uniform::<Vec<_>>(&g)));
}

benchmark_group!(sampling, lt);
benchmark_main!(sampling);
