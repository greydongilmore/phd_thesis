import sys
import csv
import nibabel as nib
import numpy as np
import pickle
import h5py


def write_dh5_np(h5_name, np_array):
    """
    :param h5_name: Name of a h5 file
    :type h5_name: str
    :param np_array: Array of floats
    :type np_array: np.array
    """

    with h5py.File(h5_name, 'w') as h5_file:
        h5_file.create_dataset('np_array', data=np_array)


def plane_intersect(a, b):
    """
    a, b   4-tuples/lists
           Ax + By +Cz + D = 0
           A,B,C,D in order

    output: 2 points on line of intersection, np.arrays, shape (3,)
    """
    a_vec, b_vec = np.array(a[:3]), np.array(b[:3])

    aXb_vec = np.cross(a_vec, b_vec)

    A = np.array([a_vec, b_vec, aXb_vec])
    d = np.array([-a[3], -b[3], 0.]).reshape(3, 1)

    # could add np.linalg.det(A) == 0 test to prevent linalg.solve throwing error

    p_inter = np.linalg.solve(A, d).T

    return p_inter[0], (p_inter + aXb_vec)[0]


def kron(a, b):
    a_s = []
    b_s = []
    if (a.shape == (3,)) & (b.shape == (3,)):
        a_s = [1, 3]
        b_s = [3, 1]

    A = np.reshape(a, (1, a_s[0], 1, a_s[1]))
    B = np.reshape(b, (b_s[0], 1, b_s[1], 1))
    K = np.reshape(A * B, [a_s[0] * a_s[1], b_s[0] * b_s[1]])
    return K


def rotate_axis(axis, degree):
    u = axis / np.linalg.norm(axis)
    m = np.eye(4)
    cosA = np.cos(degree)
    sinA = np.sin(degree)

    tmp = np.eye(4)
    kr = kron(u, u.transpose())
    tmp[0:3, 0:3] = cosA * np.eye(3) + (1 - cosA) * kr + sinA * np.array(
        [[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]]
    )
    m = np.dot(m, tmp)
    return m


def translate_p(p):
    m = np.eye(4)
    m[0, 3] = p[0]
    m[1, 3] = p[1]
    m[2, 3] = p[2]
    return m


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    inp = sys.argv[1]
    features = sys.argv[2]
    outp = sys.argv[3]
    outp_mat = sys.argv[4]

    tmp_names = []
    tmp_xyz = []
    # read CSV
    i = 0
    with open(features, newline='') as csvfile:

        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            print(', '.join(row))
            if i > 2:
                tmp_xyz.append(row[1:4])
                tmp_names.append(row[11].lower())
            i = i + 1


    nif = nib.load(inp)
    transform = nif.affine
    inv_transf = np.linalg.inv(transform)
    ac_i = tmp_names.index('ac')
    pc_i = tmp_names.index('pc')
    m1_i = tmp_names.index('m1')
    m2_i = tmp_names.index('m2')
    m3_i = tmp_names.index('m3')
    m4_i = tmp_names.index('m4')

    ac = np.array([float(x) for x in tmp_xyz[ac_i]] + [1])
    pc = np.array([float(x) for x in tmp_xyz[pc_i]] + [1])

    m1 = np.array([float(x) for x in tmp_xyz[m1_i]])
    m2 = np.array([float(x) for x in tmp_xyz[m2_i]])
    m3 = np.array([float(x) for x in tmp_xyz[m3_i]])
    m4 = np.array([float(x) for x in tmp_xyz[m4_i]])

    mid = (m1 + m2 + m3 + m4)/4
    mid = np.array( mid.tolist() + [1])
    a1 = ac - pc
    a2 = mid - pc
    a1 = a1[0:3]
    a2 = a2[0:3]
    a_cross = np.cross(a1, a2)


    mni_a = np.array([0, 0, 0])  # pc
    mni_b = np.array([0, 0, 25])  # mid
    mni_c = np.array([0, 25, 0])  # ac
    mni_a1 = mni_c - mni_a
    mni_a2 = mni_b - mni_a
    mni_across = np.cross(mni_a1, mni_a2)


    d1 = -np.dot(a_cross, ac[:3])
    d2 = -np.dot(mni_across, mni_b)
    #caclulate intersection of data MNI coords and MNI origin PLANE
    res_intr = plane_intersect(list(a_cross) + [d1], list(mni_across) + [d2])

    p = res_intr[0]
    N = (res_intr[1] - res_intr[0]) / np.linalg.norm(res_intr[1] - res_intr[0])
    #angle between two planes (MNI coords and MNI origin PLANE)
    alpha = np.arccos(np.dot(mni_across, a_cross) / (np.linalg.norm(mni_across) * np.linalg.norm(a_cross)))
    if (alpha > np.pi / 2):
        alpha = 0 - (np.pi - alpha)
    print(alpha)

    rotA = rotate_axis(N, alpha)

    mov_b = translate_p(-p)
    mov_f = translate_p(p)

    combined_aff = np.dot(mov_f, np.dot(rotA, mov_b))
    # rotate AC PC line from file to atlsas

    ac_m = np.dot(combined_aff, ac)
    pc_m = np.dot(combined_aff, pc)
    mid_m = np.dot(combined_aff, mid)

    Nr = np.cross(ac_m[:3] - mid_m[:3], pc_m[:3] - mid_m[:3])
    Nr = Nr / np.linalg.norm(Nr)

    alpha = np.arccos(np.dot(mni_c - mni_a, ac_m[:3] - pc_m[:3]) / (
            np.linalg.norm(mni_c - mni_a) * np.linalg.norm(ac_m[:3] - pc_m[:3])));
    print(alpha)

    if (alpha > np.pi / 2):
        alpha = (np.pi - alpha)
    rot_a1 = np.dot(translate_p(pc_m), np.dot(rotate_axis(Nr, alpha), translate_p(-pc_m)))
    # pick angle
    ac = np.array([0, 25, 0, 1])
    pc = np.array([0, 0, 0, 1])
    ac_new = np.dot(rot_a1, ac_m)
    pc_new = np.dot(rot_a1, pc_m)
    ac_1 = ac_new - pc_new
    l1 = np.linalg.norm(ac_1 - ac)
    rot_a2 = np.dot(translate_p(pc_m), np.dot(rotate_axis(Nr, -alpha), translate_p(-pc_m)))
    ac_new2 = np.dot(rot_a2, ac_m)
    pc_new2 = np.dot(rot_a2, pc_m)
    ac_2 = ac_new2 - pc_new2
    l2 = np.linalg.norm(ac_2 - ac)
    if l1>l2:
        rot_a1 = rot_a2

    tr_res = np.dot(rot_a1, np.dot(combined_aff, transform))
    #tr_res = np.dot(combined_aff,transform)

    #write pickle file
    f = open(outp_mat,'wb')
    pickle.dump(np.dot(rot_a1,combined_aff),f)
    # write_dh5_np(h5_name=outp,np_array=np.dot(rot_a1,rotA))
    # write_dh5_np(h5_name=outp, np_array=tr_res)
    nif2 = nib.Nifti1Image(nif.get_fdata(), tr_res)
    nib.save(nif2, outp)
    # Initialize the layout
    print(1222)

    # Print some basic information about the layout
