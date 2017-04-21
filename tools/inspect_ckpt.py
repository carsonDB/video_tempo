import sys
import tensorflow as tf

list_variables = tf.contrib.framework.list_variables

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("python inspect_ckpt.py"
              " model.ckpt")
        sys.exit()

    ckpt_path = sys.argv[-1]
    for var in list_variables(ckpt_path):
        print(var)
