import numpy as np


def convert_to_pcd(pc, output_file):
    header = [
        "VERSION .7\n",
        "FIELDS x y z\n",
        "SIZE 4 4 4\n",
        "TYPE F F F\n",
        "COUNT 1 1 1\n",
        "WIDTH 1024\n",
        "HEIGHT 768\n",
        "VIEWPOINT 0 0 0 1 0 0 0\n",
        "POINTS 786432\n",
        "DATA ascii\n"
    ]
    file = open(output_file, mode="w")
    file.writelines(header)
    for i in np.arange(pc.shape[0]):
        file.write("%.5f %.5f %.5f\n" % (pc[i, 0], pc[i, 1], pc[i, 2]))
        if i % 10 == 0:
            file.flush()

    file.close()


