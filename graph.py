from PIL import Image
import tensorflow as tf
import numpy as np

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

if __name__ == '__main__':

    # We use our "load_graph" function
    graph = load_graph("C:/Users/asus/OneDrive/Workspaces/VGG_Face_Study/features.pb")

    # # We can verify that we can access the list of operations in the graph
    # for op in graph.get_operations():
        # print(op.name)
        # # prefix/Placeholder/inputs_placeholder
        # # ...
        # # prefix/Accuracy/predictions
        
    # We access the input and output nodes 
    x = graph.get_tensor_by_name('prefix/permute_1_input:0')
    y = graph.get_tensor_by_name('prefix/flatten_1/Reshape:0')
        
    with tf.Session(graph=graph) as sess:
        y_out = sess.run(y, feed_dict={
            x: np.zeros((1, 224, 224, 3)) # replace np.zeros by np.array convert from the cropped image
        })

        print(y_out)
        print(np.shape(y_out))
