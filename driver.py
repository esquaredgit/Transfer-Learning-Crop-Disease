
import model
import fromIN
import fromScratch
import s2IN



# S2-ImageNet
e = 5
r = 0.0001
config = "E-"+str(e)+", LR-"+str(r)
print("\n\S2-ImageNet: "+config+"\n\n")
s2IN.s2ImageNet(e,r)

# S2-ImageNet
e = 10
r = 0.01
config = "E-"+str(e)+", LR-"+str(r)
print("\n\S2-ImageNet: "+config+"\n\n")
s2IN.s2ImageNet(e,r)

# S2-ImageNet
e = 10
r = 0.001
config = "E-"+str(e)+", LR-"+str(r)
print("\n\S2-ImageNet: "+config+"\n\n")
s2IN.s2ImageNet(e,r)

# S2-ImageNet
e = 10
r = 0.0001
config = "E-"+str(e)+", LR-"+str(r)
print("\n\S2-ImageNet: "+config+"\n\n")
s2IN.s2ImageNet(e,r)





