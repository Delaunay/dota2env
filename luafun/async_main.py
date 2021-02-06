"""Async is a more complex implementation"""

import logging
import time 

from luafun.dotaenv import Dota2Env
from luafun.observations import Entry
from luafun.model.actor_critic import ActorCritic, BaseActorCritic

log = logging.getLogger(__name__)


def play(obs_send, weight_recv):
    """Run the Observation Generation
    
    Parameters
    ----------
    obs_send:
        client that send the observation data to the training loop
    """
    game = Dota2Env('F:/SteamLibrary/steamapps/common/dota 2 beta/', False)
    deadline = game.deadline    

    objective = # ...
    model: ActorCritic = BaseActorCritic()

    w = weight_recv.recv()
    model.critic = None
    model.load_actor(w)

    running_reward = 0
    with game:
        # Acquire initial state
        state = game.state()

        while game.is_running():
            # Play the Game
            s = time.now()
            action = model.act(state)
            game.send_message(action)
            e = time.now()

            if e - s > deadline:
                log.debug(f"deadline unmet {e -s}")

            # this returns the last
            result_state = game.state()
            # --- 

            # Prepare the data for training
            done   = game.is_running()
            training_entry = Entry(
                action,
                state,
                done,
                result_state)
            obs_queue.send(training_entry)
            # --

            # 
            state = result_state
            running_reward += reward()

            if False:
                print(f'Reward: {running_reward:.4f}')

            # update the model
            w = weight_recv.recv()
            if w is not None:
                model.load_actor(w)

        # Wait for everything to close
        game.wait()


def train(obs_receive, weight_send):
    """Run PPO as we receive observation from playing
    
    Parameters
    ----------
    obs_receive:
        client that receive the observation data to the play loop

    Notes
    -----
    when training using multiple games in parallel the optimizer needs to be initialized per game?
    """
    policy: ActorCritic = BaseActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr, betas=betas)
    dataset = # ...
    eps_clip = # ...
    loss = nn.MSELoss()

    while True:
        new_obs = obs_receive.recv()

        if new_obs is None:
            break

        dataset.append(new_obs)
        
        running_loss = 0
        for entry in Sampler(dataset):
            state = entry.state
            action = entry.action
            # 16 Time steps for the LSTM


            # Compute reward HERE
            # ...

            # Get the gradient for the action
            logprobs, state_values, dist_entropy = policy.eval(state, action)
            
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # This is regular advantage we need GAE
            # Q: rewards
            # V: Epexted value
            # A = Q - V
            #
            # GAE is the expoentially weighted average of k steps estimators
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * loss(state_values, rewards) - 0.01 * dist_entropy
            loss = loss.mean()

            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.detach()

        # Push the new model
        # ...
        # ---
        weight_send.send(policy.actor_weights())

        if False:
            print(f'Loss: {running_loss:.4f}')



def main():
    logging.basicConfig(level=logging.DEBUG)




    print('Done')
