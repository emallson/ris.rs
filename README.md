# ris.rs -- Reverse Influence Sampling Iterators in Rust
[![Build Status](https://travis-ci.org/emallson/ris.rs.svg?branch=master)](https://travis-ci.org/emallson/ris.rs)
```toml
[dependencies]
ris = { version = "0.1", git = "https://github.com/emallson/ris.rs.git" }
```

This crate includes a set of iterators for common Reverse Influence Sampling methods. For information on RIS, see [this paper](https://arxiv.org/abs/1212.0884). The general usage pattern is:

```rust
// `g` is a petgraph::Graph<_, f32>
// for a uniformly chosen source node
let sample: Vec<NodeIndex> = IC::new_uniform(&g).collect();
// for a pre-selected source node `src`
let sample: Vec<NodeIndex> = IC::new(&g, src).collect();
```

Included models are:

- Influence Cascade (IC)
- Linear Threshold (LT)

## License
Copyright 2017 J. David Smith.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
