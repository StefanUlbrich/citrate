# Potpourri

**Warning: Still under heavy construction**

A package for mixture models and other models that can be learned with
the Expectation [Maximization (EM) algorithm](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm).

* [x] NDArray backend for mixture models
* [x] MVP: Gaussian Mixture Model (GMM) working, passes basic test
* [ ] Performance benchmark: Compare GMM to sklearn
* [ ] Prototype of federated computation of EM (using [ractor](https://github.com/slawlor/ractor) or [coerce](https://github.com/LeonHartley/Coerce-rs))
* Additional models
  * [ ] Hidden Markov Models
  * [ ] [Kalman filter](https://news.ycombinator.com/item?id=36971975)
  * [ ] Additional distributions for mixture models
* ...

## Development

```sh
sudo apt install libfontconfig-dev libopenblas-dev # on ubuntu
cargo run --package potpourri --example generate_data --features ndarray
```