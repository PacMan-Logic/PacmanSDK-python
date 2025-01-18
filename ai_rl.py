from core.gamedata import GameState
from core.GymEnvironment import PacmanEnv
from model import *
from train import state_dict_to_tensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PacmanAI:
    def __init__(self, device=device):
        self.device = device
        self.pacman_net = PacmanNet(4, 5, 40)
        self.pacman_net.load_state_dict(torch.load("pacman.pth"))
        self.pacman_net.to(self.device)
        self.pacman_net.eval()

    def __call__(self, game_state: GameState):
        state = game_state.gamestate_to_statedict()
        state_tensor, extra = state_dict_to_tensor(state)
        with torch.no_grad():
            op = (
                self.pacman_net(state_tensor.to(self.device), extra.to(self.device))
                .argmax(1)
                .cpu()
            )
        return [op.item()]


if __name__ == "__main__":
    ai = PacmanAI()
    env = PacmanEnv()
    env.reset()
    state = env.game_state()

    out = ai(state)
    print(out)
