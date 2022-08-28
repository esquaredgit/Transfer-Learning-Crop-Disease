import model
import fromIN
import fromScratch
import tensorflow as tf



with tf.device("/gpu:0"):
    e1 = 5
    r1 = 0.0001
    config = "E-"+str(e1)+", LR-"+str(r1)
    print("\n\nFROM SCRATCH: "+config+"\n\n")
    fromScratch.fromScratch(e1, r1)
    config = "E-"+str(e1)+", LR-"+str(r1)
    print("\n\nFROM ImageNet: "+config+"\n\n")
    fromIN.fromIN(e1,r1)
    config = "E-"+str(e1)+", LR-"+str(r1)
    print("\n\nNORMAL SEQUENCE: "+config+"\n\n")
    model.normSequence(e1,r1)


with tf.device("/gpu:1"):
    e2 = 15
    r2 = 0.01
    config = "E-"+str(e2)+", LR-"+str(r2)
    print("\n\nFROM SCRATCH: "+config+"\n\n")
    fromScratch.fromScratch(e2, r2)
    print("\n\nFROM ImageNet: "+config+"\n\n")
    fromIN.fromIN(e2,r2)

with tf.device("/gpu:2"):
    e3 = 15
    r3 = 0.001
    config = "E-"+str(e3)+", LR-"+str(r3)
    print("\n\nFROM SCRATCH: "+config+"\n\n")
    fromScratch.fromScratch(e3, r3)
    config = "E-"+str(e3)+", LR-"+str(r3)
    print("\n\nFROM ImageNet: "+config+"\n\n")
    fromIN.fromIN(e3,r3)
    config = "E-"+str(e3)+", LR-"+str(r3)
    print("\n\nNORMAL SEQUENCE: "+config+"\n\n")
    model.normSequence(e3,r3)

with tf.device("/gpu:3"):
    e4=15
    r4 = 0.0001
    config = "E-"+str(e4)+", LR-"+str(r4)
    print("\n\nFROM SCRATCH: "+config+"\n\n")
    fromScratch.fromScratch(e4, r4)
    config = "E-"+str(e4)+", LR-"+str(r4)
    print("\n\nFROM ImageNet: "+config+"\n\n")
    fromIN.fromIN(e4,r4)
    config = "E-"+str(e4)+", LR-"+str(r4)
    print("\n\nNORMAL SEQUENCE: "+config+"\n\n")
    model.normSequence(e4,r4)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    # Run the op.
    sess.run()