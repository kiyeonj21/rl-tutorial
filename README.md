# Reinforcement Learning Tutorial

## Installation
Install all dependencies listed in requirements.txt
## Run

(1) Grid World (tabular mode) - value iteration, policy iteration, first visit Monte Carlo, every visit Monte Carlo,\
Sarsa, Td(0), Q learning, double Q learning
 
``` python
cd ./experiments
python gridworld_policyiteration.py
python gridworld_valueiteration.py
python gridworld_firstvisit_mc.py
python gridworld_firstvisit_mc_es.py
python gridworld_firstvisit_mc_greedy.py
python gridworld_everyvisit_mc.py
python gridworld_everyvisit_mc_es.py
python gridworld_q_learning.py
python gridworld_sarsa.py
python gridworld_td0.py
python gridworld_double_q_learning.py
cd ../
```

(2) Cartpole - DQN, PGD, REINFORCE, Arctic Critic
``` python
cd ./experiments
python cartpole_dqn.py
python cartpole_pgd.py
python cartpole_reinforce.py
python cartpole_arcticcritic.py
cd ../
```

## Reference
Book \
(1) [Reinforcement Learning: An Introduction, R. Sutton and A. Barto](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf) \
(2) [Reinforcement Learning and Optimal Control](https://www.amazon.com/Reinforcement-Learning-Optimal-Control-Bertsekas/dp/1886529396/ref=sr_1_9?dchild=1&keywords=reinforcement+learning&qid=1614358839&sr=8-9) \
(3) [Dynamic Programming and Optimal Control (2 Vol Set)](https://www.amazon.com/Dynamic-Programming-Optimal-Control-Vol/dp/1886529086/ref=pd_bxgy_img_2/140-2898019-6489713?_encoding=UTF8&pd_rd_i=1886529086&pd_rd_r=bd4cf0f1-58ca-401c-82e4-aca100f11005&pd_rd_w=FktV5&pd_rd_wg=1CvTp&pf_rd_p=f325d01c-4658-4593-be83-3e12ca663f0e&pf_rd_r=W41D2S4NZN6YT83ZB03B&psc=1&refRID=W41D2S4NZN6YT83ZB03B)

Lecture \
(1) [Deep Reinforcement Learning](http://rail.eecs.berkeley.edu/deeprlcourse/) \
(2) [CS234: Reinforcement Learning](https://web.stanford.edu/class/cs234/)