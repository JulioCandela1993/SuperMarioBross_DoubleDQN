# Training a Reinforcement Learning Agent with Double Deep Q Network for Super Mario Bros

The purpose of this research is to implement the current state of
the art in Double DQN and evaluate its performance in a famous
game such as Super Mario Bros. We will be using the gym-super-mario-bros environment, which is built on top
of the OpenAI gym library. A state of the game is represented by
a list of 4 consecutive frames of 240 × 256 × 3 (height × width ×
3 channels RGB). The size of the action space in this game is 256
which corresponds to 256 possible actions of the character.

The research will show the benefits of using CNN in complex and high-dimensional problems in
Artificial Intelligence. Then, the Double DQN will be presented to
deal with some of the deficiencies of CNN in DQN so that the agent
can learn better policies and maximize the rewards.