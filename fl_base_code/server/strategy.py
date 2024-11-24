from flwr.server.strategy import FedAvg

class CustomStrategy(FedAvg):
    def __init__(self):
        super().__init__()

    def configure_fit(self, rnd, parameters, client_manager):
        config = {"epochs": 1}
        return super().configure_fit(rnd, parameters, client_manager, config=config)

    def configure_evaluate(self, rnd, parameters, client_manager):
        return super().configure_evaluate(rnd, parameters, client_manager)
