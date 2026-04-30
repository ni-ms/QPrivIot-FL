
# QPrivIot-FL: A Flower / PyTorch app

## Install dependencies and project

The dependencies are listed in the `pyproject.toml` and you can install them as follows:

```bash
pip install -e .
```

> **Tip:** Your `pyproject.toml` file can define more than just the dependencies of your Flower app. You can also use it to specify hyperparameters for your runs and control which Flower Runtime is used. By default, it uses the Simulation Runtime, but you can switch to the Deployment Runtime when needed.
> Learn more in the [TOML configuration guide](https://flower.ai/docs/framework/how-to-configure-pyproject-toml.html).

## Run with the Simulation Engine

In the `QPrivIot-FL` directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```

To run with specific configurations (e.g., deterministic seeding, alpha sweep):

```bash
flwr run . --run-config 'seed=1337 num-server-rounds=50 dataset="cifar10" dirichlet-alpha=0.3'
```

## Running Tests

To run the unit and integration tests:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run fast tests
pytest tests/ -m "not slow"

# Run all tests (including those that download datasets)
pytest tests/
```

## Run with the Deployment Engine

Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be interested in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

You can run Flower on Docker too! Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.

## Resources

- Flower website: [flower.ai](https://flower.ai/)
- Check the documentation: [flower.ai/docs](https://flower.ai/docs/)
- Give Flower a ⭐️ on GitHub: [GitHub](https://github.com/adap/flower)
- Join the Flower community!
  - [Flower Slack](https://flower.ai/join-slack/)
  - [Flower Discuss](https://discuss.flower.ai/)


## Experiments

```bash
# Baseline (No DP)
flwr run . --run-config 'use-dp=false'

# Uniform DP
flwr run . --run-config 'use-dp=true use-adaptive-dp=false'

# AdaPriv (Adaptive DP)
flwr run . --run-config 'use-dp=true use-adaptive-dp=true'
```