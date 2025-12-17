
test-all:
	python code/AIGame.py -n 10000 --eval
	python code/benchmarkAgent.py -n 20 --weights weights/catan-dqn-5310.weights.h5

test-q-learning:
	python code/AIGame.py -n 10000 --eval

test-rl-agent:
	python code/benchmarkAgent.py -n 20 --weights weights/catan-dqn-5310.weights.h5

test-rl-agent-long:
	python code/benchmarkAgent.py -n 1000 --weights weights/catan-dqn-5310.weights.h5

