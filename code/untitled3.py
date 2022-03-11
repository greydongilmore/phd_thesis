from scipy.interpolate import BPoly
import transforms3d


def applyTransformToPoints( transform, points, reverse=False):
	transformMatrix = vtk.vtkGeneralTransform()
	if reverse:
		slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, transform, transformMatrix)
	else:
		slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(transform, None, transformMatrix)
	finalPoints = transformMatrix.TransformPoint(points)
	return np.array(finalPoints)

fc=[0]*3
getNode('frame_center').GetNthControlPointPositionWorld(0, fc)
fc=np.array(fc)
ac=[0]*3
getNode('acpc').GetNthControlPointPositionWorld(0, ac)
ac=np.array(ac)
pc=[0]*3
getNode('acpc').GetNthControlPointPositionWorld(1, pc)
pc=np.array(pc)
mid=[0]*3
getNode('midline').GetNthControlPointPositionWorld(0, mid)
mid=np.array(mid)


coordsToFrame = np.array([
	[-1, 0, 0, (100+fc[0])],
	[ 0, 1, 0, (100+fc[1])],
	[ 0, 0,-1, (100+fc[2])],
	[ 0, 0, 0,   1]
])





# Transposition of image-defined trajectories into arc-quadrant centered stereotactic systems
# L.Zamorano, A.Martinez-Coll, and M.Dujovny
# Acta Neurochirurgica, Suppl. 46,109-111(1989)
#

Target=np.array([105.90,95.00,98.00])
Entry=np.array([127.40,145.00,31.70])

#Distance

Rx = Entry[0] - Target[0]
Ry = Entry[1] - Target[1]
Rz = Entry[2] - Target[2]

R1 = np.sqrt(math.pow(Ry, 2) + math.pow(Rz, 2))
R2 = np.sqrt(math.pow(Rx, 2) + math.pow(Rz, 2))

# B (angle in Y-Z plane, anterior angle, D-angle):
B = math.degrees(math.acos(Ry/R1))

# A (angle in X-Z plane, lateral angle, E-angle):
A = math.degrees(math.acos(Rx/R2))


def unit(v):
	x,y,z = v
	mag = np.linalg.norm(v)
	return (x/mag, y/mag, z/mag)

def scale(v,sc):
	x,y,z = v
	return np.array([x * sc, y * sc, z * sc])

def pnt2line(pnt, start, end):
	line_vec = start-end
	pnt_vec = start-pnt
	line_len = np.linalg.norm(line_vec)
	line_unitvec = unit(line_vec)
	pnt_vec_scaled = scale(pnt_vec, 1.0/line_len)
	t = line_unitvec @ pnt_vec_scaled
	if t < 0.0:
		t = 0.0
	elif t > 1.0:
		t = 1.0
	nearest = scale(line_vec, t)
	dist = np.linalg.norm(nearest-pnt_vec)
	nearest =nearest=start
	return (dist, nearest)

def lineseg_dists(p, a, b):
	"""Cartesian distance from point to line segment

	Edited to support arguments as series, from:
	https://stackoverflow.com/a/54442561/11208892

	Args:
		- p: np.array of single point, shape (2,) or 2D array, shape (x, 2)
		- a: np.array of shape (x, 2)
		- b: np.array of shape (x, 2)
	"""
	# normalized tangent vectors
	d_ba = b - a
	d = np.divide(d_ba, (np.hypot(d_ba[:, 0], d_ba[:, 1]).reshape(-1, 1)))
	# signed parallel distance components
	# rowwise dot products of 2D vectors
	s = np.multiply(a - p, d).sum(axis=1)
	t = np.multiply(p - b, d).sum(axis=1)
	# clamped parallel distance
	h = np.maximum.reduce([s, t, np.zeros(len(s))])
	# perpendicular distance component
	# rowwise cross products of 2D vectors  
	d_pa = p - a
	c = d_pa[:, 0] * d[:, 1] - d_pa[:, 1] * d[:, 0]
	return np.hypot(h, c)


def distance_from_line(p, p_1, p_2):
	"""
	Computes distance of a point p_3, from a line defined by p and p_1.
	See `here <https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line>`_.
	:return: euclidean distance
	"""
	# I want notation same as wikipedia page, so disabling warning.
	# pylint: disable=invalid-name
	n = p_2 - p_1
	n = n / np.linalg.norm(n)
	a_minus_p = p_1- p
	vector_to_line = a_minus_p - (np.dot(a_minus_p, n) * n)
	distance = np.linalg.norm(vector_to_line)
	return distance



def shMatrixRotFromTwoSystems(from0,from1,from2,to0,to1,to2,transformationMatrix):
	sys_matrix=vtk.vtkMatrix4x4()
	x1 = from1 - from0
	if len(x1)== 0.0:
		print("shMatrixRotFromTwoSystems: from1 - from0 == 0.0 return ERROR\n")
		return
	else:
		x1=x1/np.sqrt(np.sum(x1**2))
	y1 = from2 - from0
	if len(y1)== 0.0:
		print("shMatrixRotFromTwoSystems: from2 - from0 == 0.0 return ERROR\n")
		return
	else:
		y1=y1/np.sqrt(np.sum(y1**2))
	x2 = to1 - to0
	if len(x2)== 0.0:
		print("shMatrixRotFromTwoSystems: to1 - to0 == 0.0 return ERROR\n")
		return
	else:
		x2=x2/np.sqrt(np.sum(x2**2))
	y2 = to2 - to0
	if len(y2)== 0.0:
		print("shMatrixRotFromTwoSystems: to2 - to0 == 0.0 return ERROR\n")
		return
	else:
		y2=y2/np.sqrt(np.sum(y2**2))
	cos1 = x1 @ y1
	cos2 = x2 @ y2
	if (abs(1.0 - cos1) <= 0.000001) & (abs(1.0 - cos2) <= 0.000001):
		sys_matrix.SetElement(3, 0, to0[0] - from0[0])
		sys_matrix.SetElement(3, 1, to0[1] - from0[1])
		sys_matrix.SetElement(3, 2, to0[2] - from0[2])
		transformationMatrix.SetMatrixTransformToParent(sys_matrix)
	if abs(cos1 - cos2) > 0.08:
		sys_matrix.SetElement(3, 0, to0[0] - from0[0])
		sys_matrix.SetElement(3, 1, to0[1] - from0[1])
		sys_matrix.SetElement(3, 2, to0[2] - from0[2])
		transformationMatrix.SetMatrixTransformToParent(sys_matrix)
	z1 = np.cross(x1,y1)
	z1 = z1/np.sqrt(np.sum(z1**2))
	y1 = np.cross(z1,x1)
	z2 = np.cross(x2,y2)
	z2 = z2/np.sqrt(np.sum(z2**2))
	y2 = np.cross(z2,x2)
	detxx = (y1[1] * z1[2] - z1[1] * y1[2])
	detxy = -(y1[0] * z1[2] - z1[0] * y1[2])
	detxz = (y1[0] * z1[1] - z1[0] * y1[1])
	detyx = -(x1[1] * z1[2] - z1[1] * x1[2])
	detyy = (x1[0] * z1[2] - z1[0] * x1[2])
	detyz = -(x1[0] * z1[1] - z1[0] * x1[1])
	detzx = (x1[1] * y1[2] - y1[1] * x1[2])
	detzy = -(x1[0] * y1[2] - y1[0] * x1[2])
	detzz = (x1[0] * y1[1] - y1[0] * x1[1])
	txx = x2[0] * detxx + y2[0] * detyx + z2[0] * detzx
	txy = x2[0] * detxy + y2[0] * detyy + z2[0] * detzy
	txz = x2[0] * detxz + y2[0] * detyz + z2[0] * detzz
	tyx = x2[1] * detxx + y2[1] * detyx + z2[1] * detzx
	tyy = x2[1] * detxy + y2[1] * detyy + z2[1] * detzy
	tyz = x2[1] * detxz + y2[1] * detyz + z2[1] * detzz
	tzx = x2[2] * detxx + y2[2] * detyx + z2[2] * detzx
	tzy = x2[2] * detxy + y2[2] * detyy + z2[2] * detzy
	tzz = x2[2] * detxz + y2[2] * detyz + z2[2] * detzz
	# set transformation
	dx1 = from0[0]
	dy1 = from0[1]
	dz1 = from0[2]
	dx2 = to0[0]
	dy2 = to0[1]
	dz2 = to0[2]
	sys_matrix.SetElement(0, 0, txx)
	sys_matrix.SetElement(1, 0, txy)
	sys_matrix.SetElement(2, 0, txz)
	sys_matrix.SetElement(0, 1, tyx)
	sys_matrix.SetElement(1, 1, tyy)
	sys_matrix.SetElement(2, 1, tyz)
	sys_matrix.SetElement(0, 2, tzx)
	sys_matrix.SetElement(1, 2, tzy)
	sys_matrix.SetElement(2, 2, tzz)
	sys_matrix.SetElement(3, 0, dx2 - txx * dx1 - txy * dy1 - txz * dz1)
	sys_matrix.SetElement(3, 1, dy2 - tyx * dx1 - tyy * dy1 - tyz * dz1)
	sys_matrix.SetElement(3, 2, dz2 - tzx * dx1 - tzy * dy1 - tzz * dz1)
	transformationMatrix.SetMatrixTransformToParent(sys_matrix)
	return transformationMatrix


def entryAndTargetToAnteriorRadAngle(entry,target):
	#Distance
	Rx = target[0] - entry[0]
	Ry = target[1] - entry[1]
	Rz = target[2] - entry[2]
	anteriorAngle=None
	if math.isclose(Rx, 0.0, rel_tol=1e-09) & math.isclose(Ry, 0.0, rel_tol=1e-09) & math.isclose(Rz, 0.0, rel_tol=1e-09):
		return
	
	R1 = np.sqrt(math.pow(Ry, 2) + math.pow(Rz, 2))
	anteriorAngle = math.acos(Ry / R1)
	anteriorAngle -= np.pi * 0.5
	return anteriorAngle

from shapely.geometry import LineString as shLs
from shapely.geometry import Point as shPt

transformInternalToAcPcMatrix = slicer.vtkMRMLLinearTransformNode()
transformInternalToAcPcMatrix.SetName('transformInternalToAcPcMatrix')
slicer.mrmlScene.AddNode(transformInternalToAcPcMatrix)

acInternalCoords=[0]*3
getNode('acpc').GetNthControlPointPositionWorld(0, acInternalCoords)
acInternalCoords=np.array(acInternalCoords)
pcInternalCoords=[0]*3
getNode('acpc').GetNthControlPointPositionWorld(1, pcInternalCoords)
pcInternalCoords=np.array(pcInternalCoords)
mpPointInternalCoords=[0]*3
getNode('midline').GetNthControlPointPositionWorld(0, mpPointInternalCoords)
mpPointInternalCoords=np.array(mpPointInternalCoords)


acPcMidPoint=np.array([0.0,0.0,0.0])
internalAcToPcVec = acInternalCoords + (acInternalCoords - pcInternalCoords)
internalMidPoint = (acInternalCoords + pcInternalCoords)/2
acPcAcToPcVec =np.array([0.0,float(mag_vec(pcInternalCoords,acInternalCoords)),0.0])

l = shLs([ acInternalCoords, pcInternalCoords])
dist=shPt(mpPointInternalCoords).distance(l)

internalAcPcClosestPointToMpp= np.array(l.interpolate(dist).coords[0])

internalLinePerpendicular = mpPointInternalCoords- internalAcPcClosestPointToMpp
internalMcpPerpendicular = internalMidPoint + internalLinePerpendicular
acPcAcPerpendicular = np.array([0.0, 0.0, np.linalg.norm(internalLinePerpendicular)])


mppLineCoordinate3=np.vstack((acInternalCoords,pcInternalCoords,mpPointInternalCoords))
mppLineCoordinate3 = points.mean(axis=1)
planeNormal = np.cross(points[:,1] - points[:,0], points[:,2] - points[:,0])
planeX = points[:,1] - points[:,0]


M=shMatrixRotFromTwoSystems(
	internalMidPoint,
	internalAcToPcVec,
	internalMcpPerpendicular,
	acPcMidPoint,
	acPcAcToPcVec,
	acPcAcPerpendicular,
	transformInternalToAcPcMatrix
)

matrixFromWorld = vtk.vtkMatrix4x4()
transformInternalToAcPcMatrix.GetMatrixTransformFromWorld(matrixFromWorld)
internalAcToPcVec_world = np.append(internalAcToPcVec,1).tolist()
internalAcToPcVec_local = matrixFromWorld.MultiplyPoint(internalAcToPcVec_world)
internalAcToPcVec_local




volRasToijk = vtk.vtkMatrix4x4()
getNode('sub-P070_space-leksellg_desc-rigid_acq-Frame_ct').GetRASToIJKMatrix(volRasToijk)

niftiTranslation=np.c_[np.zeros((4,3)),np.append(slicer.util.arrayFromVTKMatrix(volRasToijk)[:3, 3],1)]
niftiRotation=np.vstack((np.c_[slicer.util.arrayFromVTKMatrix(volRasToijk)[:3, :3],np.zeros((3,1))],np.array([0,0,0,1])))
niftiScaleFactor=np.diag((np.array([
	np.linalg.norm(slicer.util.arrayFromVTKMatrix(volRasToijk)[:3, 0]),
	np.linalg.norm(slicer.util.arrayFromVTKMatrix(volRasToijk)[:3, 1]),
	np.linalg.norm(slicer.util.arrayFromVTKMatrix(volRasToijk)[:3, 2]),1])
))

niftiScaleOrientation=


reader = vtk.vtkNIFTIImageReader()
reader.SetFileName(getNode('sub-P070_T1w').GetStorageNode().GetFileName())
reader.SetTimeAsVector(True)
reader.Update()

header = reader.GetNIFTIHeader()

qFormMatrix = reader.GetQFormMatrix()
if not qFormMatrix:
	logger.debug('Warning: %s does not have a QFormMatrix - using Identity')
	qFormMatrix = vtk.vtkMatrix4x4()

spacing = reader.GetOutputDataObject(0).GetSpacing()
timeSpacing = reader.GetTimeSpacing()
nFrames = reader.GetTimeDimension()
if header.GetIntentCode() != header.IntentTimeSeries:
	intentName = header.GetIntentName()
	if not intentName:
		intentName = 'Nothing'
	logger.debug('Warning: %s does not have TimeSeries intent, instead it has \"%s\"' % (niiFileName,intentName))
	logger.debug('Trying to read as TimeSeries anyway')

units = header.GetXYZTUnits()
image->GetSpacing(spacing);
  image->GetOrigin(origin);
  image->GetExtent(extent);

spacing=getNode('sub-P070_space-leksellg_desc-rigid_acq-Frame_ct').GetSpacing()
origin=getNode('sub-P070_space-leksellg_desc-rigid_acq-Frame_ct').GetOrigin()
extent=getNode('sub-P070_space-leksellg_desc-rigid_acq-Frame_ct').GetImageData().GetExtent()

centre[0] = 0.5*(extent[0] + extent[1])*spacing[0] + origin[0]
  centre[1] = 0.5*(extent[2] + extent[3])*spacing[1] + origin[1]
  centre[2] = 0.5*(extent[4] + extent[5])*spacing[2] + origin[2]
