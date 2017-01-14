import json
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import SigmoidLayer, TanhLayer, SoftmaxLayer

f = open("threes.json", "r")
threes = f.read()
f.close()
threes = json.loads(threes)


f = open("fours.json", "r")
fours = f.read()
f.close()
fours = json.loads(fours)


f = open("threes_test.json", "r")
test_threes = f.read()
f.close()
test_threes = json.loads(test_threes)


f = open("fours_test.json", "r")
test_fours = f.read()
f.close()
test_fours = json.loads(test_fours)


INPUTS = 64
HIDDEN_NODES = 15
OUTPUTS = 1

net = buildNetwork(INPUTS, HIDDEN_NODES, HIDDEN_NODES, OUTPUTS, bias=True, outclass=SigmoidLayer)
ds = SupervisedDataSet(INPUTS, OUTPUTS)

for three in threes: #Iterate through the list of sprites
    ds.addSample(three, 0) #three is the pictograph ones and zeros. 

for four in fours: #Iterate through the list of sprites
    ds.addSample(four, 1) #three is the pictograph ones and zeros.


trainer = BackpropTrainer(net, ds)
error = 0
for i in range(0,500): #train 5000 iterations
    error = trainer.train()
    print(error)
    if i % 20 == 0:
        correct = 0
        wrong = 0
        for test in test_threes:
            output = net.activate(test)
            if output < .5: correct += 1
            else:wrong += 1
        for test in test_fours:
            output = net.activate(test)
            if output > .5: correct += 1
            else:wrong += 1
        print("-----------")
        print("Correct", correct)
        print("Incorrect", wrong)
        print("-----------")

print("Testing Threes--------------")
for test in test_threes:
    output = net.activate(test)
    if output < .5:
        print("Three Detected")
    else:
        print("Four Detected")

print("Testing Fours--------------")
for test in test_fours:
    output = net.activate(test)
    if output < .5:
        print("Three Detected")
    else:
        print("Four Detected")
#sol = net.activate(test)