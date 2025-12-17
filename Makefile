
test-all:
	python code/AIGame.py -n 10000 --eval --weights weights/QLearner.npy
	python code/benchmarkAgent.py -n 20 --weights weights/catan-dqn-5310.weights.h5

test-q-learning:
	python code/AIGame.py -n 10000 --eval --weights weights/QLearner.npy

test-rl-agent:
	python code/benchmarkAgent.py -n 20 --weights weights/catan-dqn-5310.weights.h5

test-rl-agent-long:
	python code/benchmarkAgent.py -n 1000 --weights weights/catan-dqn-5310.weights.h5

test-head-to-head:
	python code/benchmarkAgent.py -n 20 --weights weights/catan-dqn-5310.weights.h5 --opponent QLearner
