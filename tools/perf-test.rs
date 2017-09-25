extern crate docopt;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate capngraph;
extern crate ris;
extern crate vec_graph;
extern crate rand_mersenne_twister;

#[cfg_attr(rustfmt, rustfmt_skip)]
const USAGE: &str = "
Usage:
    perf-test <graph> <model> <samples> [options]
    perf-test (-h | --help)
    
Options:
    -h --help       Show this screen.
";

use ris::*;
use vec_graph::*;
use rand_mersenne_twister::mersenne;
use std::time;

#[derive(Serialize, Deserialize, Debug)]
struct Args {
    arg_graph: String,
    arg_model: Model,
    arg_samples: usize,
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone)]
enum Model {
    IC,
    LT,
}

fn main() {
    let args: Args = docopt::Docopt::new(USAGE)
        .and_then(|d| d.deserialize())
        .unwrap_or_else(|e| e.exit());

    let g = VecGraph::from_edges(capngraph::load_edges(&args.arg_graph).unwrap());
    let mut stor = Vec::with_capacity(args.arg_samples);

    let mut rng = mersenne();

    let start = time::SystemTime::now();
    for _ in 0..args.arg_samples {
        stor.push(match args.arg_model {
            Model::IC => IC::new_uniform_with::<Vec<_>, _>(&mut rng, &g),
            Model::LT => unimplemented!(),
        })
    }
    let end = time::SystemTime::now();
    let elapsed = end.duration_since(start).unwrap();
    println!("sampled {} in {} (+ {})",
             args.arg_samples,
             elapsed.as_secs(),
             elapsed.subsec_nanos());
}
