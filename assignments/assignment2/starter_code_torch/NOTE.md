

# Q2

```bash
docker run --rm -it -v `pwd`:/opt/work/ -e DISPLAY=${ip}:0 -v /tmp/.X11-unix:/tmp/.X11-unix mebusy/gym q2_schedule.py


```


## Tips

### numpy linear interpolation

```python
self.epsilon = np.interp( t , [0,self.nsteps], [self.eps_begin,self.eps_end] )
```

### action exploration

```python
return best_action if np.random.random() > self.epsilon else self.env.action_space.sample()
```


# Q3

- To improve the data efficiency and stability
    - replay buffer
    - target network （temporarily fixed θ）
- Q: DeepMind’s deep Q network (DQN) takes as input the state s and outputs a vector of size |A|, the number of actions. What is one benefit of computing the Q function as Q<sub>θ</sub>(s,·) ∈ ℝ<sup>|A|</sup>, as opposed to Q<sub>θ</sub>(s,a) ∈ ℝ?
    - A: : In order to make a Q-Learning update, we must identify the action which maximizes q̂ at the state s'.
        - If we parameterized the DQN to take both a state and action as input, we would need to execute O(|A|) forward-passes (one per legal action) in order to compute this argmax. 
        - By instead parameterizing the DQN such that it takes only a state as input and outputs the state-action values for all legal actions in that state simultaneously, we need only execute a single forward-pass per Q-Learning update.



