import os


def CT_to_frame( TP, FC):
	frame_target_X = np.diff((FC[0],TP[0]))*-1 + 100
	frame_target_Y = np.diff((FC[1],TP[1])) + 100
	frame_target_Z = np.diff((FC[2],TP[2]))*-1 + 100
	return np.r_[frame_target_X, frame_target_Y, frame_target_Z]

def frame_to_CT(TP,FC):
	frame_target_X = ((TP[0]+FC[0])) - 100
	frame_target_Y = TP[1]+FC[1]-100
	frame_target_Z = ((TP[2]+FC[2])) - 100
	return np.array([frame_target_X, frame_target_Y, frame_target_Z])

def convert_ras(point, img_obj):
	affine_rasTOijk= vtk.vtkMatrix4x4()
	img_obj.GetRASToIJKMatrix(affine_rasTOijk)
	if isinstance(point,(np.ndarray, np.generic)):
		point = point.tolist()
	position_ras = point + [1]
	position_ijk=affine_rasTOijk.MultiplyPoint(position_ras)
	
	return np.array(position_ijk[:3])

def convert_ijk(point, img_obj):
	affine_ijkTOras= vtk.vtkMatrix4x4()
	img_obj.GetIJKToRASMatrix(affine_ijkTOras)
	if isinstance(point,(np.ndarray, np.generic)):
		point = point.tolist()
	position_ijk = point + [1]
	position_ras=affine_ijkTOras.MultiplyPoint(position_ijk)
	
	return np.array(position_ras[:3])


def frame_coordinates(ac, pc, mid):
	AC_frame = ac
	PC_frame = pc
	MP_frame = mid
	#AC_frame = np.array(CT_to_frame(ac,fc))
	#PC_frame = np.array(CT_to_frame(pc,fc))
	#MP_frame = np.array(CT_to_frame(mid,fc))
	MCP_frame = np.array([(AC_frame[0]+PC_frame[0])/2, (AC_frame[1]+PC_frame[1])/2,(AC_frame[2]+PC_frame[2])/2])
	IC = AC_frame-PC_frame
	len_IC = np.sqrt((IC[0])**2+(IC[1])**2+(IC[2])**2)
	e_y = np.array([IC[0]/len_IC, IC[1]/len_IC, IC[2]/len_IC])
	w = np.array(MP_frame)-np.array(MCP_frame)
	if float(np.diff((MCP_frame[2], MP_frame[2]))) < 0:
		crossp=np.cross(w,e_y)
	else:
		crossp=np.cross(e_y,w)
	e_x = crossp/np.sqrt((crossp[0])**2+(crossp[1])**2+(crossp[2])**2)
	e_z = np.cross(e_x, e_y)
	#TP_frame = MCP_frame+ (target[0]*e_x) + (target[1]*e_y) + (target[2]*e_z)
	#standard basis
	xihat=np.array([1,0,0])
	yihat=np.array([0,1,0])
	zihat=np.array([0,0,1])
	# Rotation matrix
	riiprime = np.vstack([np.array([e_x.dot(xihat), e_x.dot(yihat), e_x.dot(zihat)]),
						np.array([e_y.dot(xihat), e_y.dot(yihat), e_y.dot(zihat)]),
						np.array([e_z.dot(xihat), e_z.dot(yihat), e_z.dot(zihat)])])
	return riiprime

def getFrameRotation(ac, pc, mid):
	pmprime = (ac + pc) / 2
	vec1 = ac - pmprime
	vec2 = mid - pmprime
	vec1Mag = np.sqrt((vec1[0] ** 2) + (vec1[1] ** 2) + (vec1[2] ** 2))
	vec2Mag = np.sqrt((vec2[0] ** 2) + (vec2[1] ** 2) + (vec2[2] ** 2))
	vec1Unit = vec1 / vec1Mag
	vec2Unit = vec2 / vec2Mag
	yihatprime = vec1Unit
	if pmprime[2] > mid[2]:
		crossp = np.cross(vec2Unit, yihatprime)
	else:
		crossp = np.cross(yihatprime, vec2Unit)
	xAxisMag = np.sqrt((crossp[0] ** 2) + (crossp[1] ** 2) + (crossp[2] ** 2))
	xihatprime = crossp / xAxisMag
	zAxis = np.cross(xihatprime, yihatprime)
	zAxisMag = np.sqrt((zAxis[0] ** 2) + (zAxis[1] ** 2) + (zAxis[2] ** 2))
	zihatprime = zAxis / zAxisMag
	xihat = np.array([1, 0, 0])
	yihat = np.array([0, 1, 0])
	zihat = np.array([0, 0, 1])
	riiprime = np.vstack([np.array([xihatprime.dot(xihat), xihatprime.dot(yihat), xihatprime.dot(zihat)]),
						 np.array([yihatprime.dot(xihat), yihatprime.dot(yihat), yihatprime.dot(zihat)]),
						 np.array([zihatprime.dot(xihat), zihatprime.dot(yihat), zihatprime.dot(zihat)])])
	return riiprime

def applyTransformToPoints(transform, points, reverse=False):
	transformMatrix = vtk.vtkGeneralTransform()
	if reverse:
		slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, transform, transformMatrix)
	else:
		slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(transform, None, transformMatrix)
	finalPoints = transformMatrix.TransformPoint(points)
	return np.array(finalPoints)

#sub-P175
entAP=np.array([-34.12,35,67.29])
tarAP=np.array([-11.09,-3.43,-3.99])
ent_frame=CT_to_frame(mcp+entAP,fc)
tar_frame=CT_to_frame(mcp+tarAP,fc)

#sub-P057
entAP=np.array([-31.52,47.86,70.89])
tarAP=np.array([-12.62,-2.13,-3.87])

#sub-P057
entAP=np.array([-33.6,39.36,61.5])
tarAP=np.array([-11.73,-3.04,-3.09])

#sub-P057
entAP=np.array([-45.21,54.05,51.95])
tarAP=np.array([-11.73,-3.04,-3.09])


ent_frame=CT_to_frame(mcp+entAP,fc)
tar_frame=CT_to_frame(mcp+tarAP,fc)

frame_to_CT(tar_frame,mcp_frame)

mcp=[0]*3
getNode('acpc').GetNthControlPointPositionWorld(2, mcp)
mcp=np.array(mcp)
ac=[0]*3
getNode('acpc').GetNthControlPointPositionWorld(0, ac)
ac=np.array(ac)
pc=[0]*3
getNode('acpc').GetNthControlPointPositionWorld(1, pc)
pc=np.array(pc)
mid=[0]*3
getNode('midline').GetNthControlPointPositionWorld(0, mid)
mid=np.array(mid)
fc=[0]*3
getNode('frame_center').GetNthControlPointPositionWorld(0, fc)
fc=np.array(fc)

crossHairNode = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLCrosshairNode')
crossHairRAS = np.array(crossHairNode.GetCrosshairRAS())

img_obj=getNode('sub-P076_ses-perisurg_acq-Frame_run-01_ct')
ac_dicom=convert_ras(ac, img_obj)
pc_dicom=convert_ras(pc, img_obj)
mid_dicom=convert_ras(mid, img_obj)
fc_dicom=convert_ras(fc, img_obj)
mcp_dicom=(ac_dicom+pc_dicom)/2

print(ac)
print(pc)
print(mid)
print(ac_dicom)
print(pc_dicom)
print(mid_dicom)


(ac_dicom-fc_dicom)*np.array(img_obj.GetSpacing())
(pc_dicom-fc_dicom)
(mid_dicom-fc_dicom)
(tar_dicom-fc_dicom)


ac_frame=CT_to_frame(ac_dicom,fc_dicom,)*np.array(img_obj.GetSpacing())
pc_frame=CT_to_frame(pc,fc_dicom)
mcp_frame=(ac_frame+pc_frame)/2
fc_frame=CT_to_frame(np.array([0,0,0]),fc)
mid_frame=CT_to_frame(mid,fc)

ac_frame
pc_frame
mid_frame
tar_frame

newCoordsEnt=applyTransformToPoints(slicer.util.getNode('acpc_transform'), crossHairRAS, reverse=True)
new_mcp=applyTransformToPoints(list(slicer.util.getNodes('*desc-affine_from-ctFrame*').values())[0], mcp, reverse=False)
new_fc=applyTransformToPoints(list(slicer.util.getNodes('*desc-affine_from-ctFrame*').values())[0], fc, reverse=False)
new_crossHairRAS=applyTransformToPoints(slicer.util.getNode('acpc_transform'), crossHairRAS-mcp, reverse=False)
applyTransformToPoints(list(slicer.util.getNodes('*desc-affine_from-ctFrame*').values())[0], new_crossHairRAS-new_mcp, reverse=True)



entAP=np.array([-45.21,54.05,51.95])
entF=np.array([132.4,151.3,23.5])

#midACPC
riiprime=getFrameRotation(ac,pc,mid)

ac_frame=CT_to_frame(ac,fc)
pc_frame=CT_to_frame(pc,fc)
mid_frame=CT_to_frame(mid,fc)
fc_frame=CT_to_frame(np.array([0,0,0]),fc)
mcp_frame=(ac_frame+pc_frame)/2


riiprime=getFrameRotation(ac_frame,pc_frame,mid_frame)

crossHairCoords=riiprime.T.dot(entAP+mcp)

riiprime=getFrameRotation(ac_frame,pc_frame,mid_frame)
crossHairCoords=riiprime.T.dot(crossHairCoords)+mcp


crossHairNode = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLCrosshairNode')
crossHairRAS = np.array(crossHairNode.GetCrosshairRAS())


riiprime=getFrameRotation(ac_frame,pc_frame,mid_frame)

riiprime.dot(entF)




currentFramecalc=riiprime.T.dot(entAP)
currentframecalcPM=((currentFramecalc))+fc_frame
currentframecalcPM

xyzpmdelta=(currentframecalcPM-mcp_frame-(mcp_frame-fc_frame))*np.array([-1,1,-1])
computeanatomicfromframe=riiprime.dot(xyzpmdelta+fc)
computeanatomicfromframe


def Targets_to_Frame(Xt, Xe):
	# unit vector from target to entry point
	Xr = np.array(Xe-Xt)
	#The distance between the points.
	dist = np.linalg.norm(Xr)
	#arc and collar
	phi=np.array([0.0,0.0])
	phi[0]=np.arccos(Xr[0]/dist)
	if Xr[1]!=0:
		phi[1]=np.arctan(Xr[2]/Xr[1])
	else:
		phi[1]=np.pi/2.0
	if phi[1]<0:
		phi[1]=np.pi+phi[1]
	return [math.degrees(phi[0]),math.degrees(phi[1])]

def rotation_matrix(pitch, roll, yaw):
	#pitch: y, roll: x, yaw: z
	pitch, roll, yaw = np.array([pitch, roll, yaw]) * np.pi / 180
	matrix_pitch = np.array([
		[np.cos(pitch), 0, np.sin(pitch)],
		[0, 1, 0],
		[-np.sin(pitch), 0, np.cos(pitch)]
	])
	matrix_roll = np.array([
		[1, 0, 0],
		[0, np.cos(roll), -np.sin(roll)],
		[0, np.sin(roll), np.cos(roll)]
	])
	matrix_yaw = np.array([
		[np.cos(yaw), -np.sin(yaw), 0],
		[np.sin(yaw), np.cos(yaw), 0],
		[0, 0, 1]
	])
	return np.dot(matrix_pitch, np.dot(matrix_roll, matrix_yaw))

entACPC=np.array([-31.52,47.82,70.89,1])
riiprime=getFrameRotation(ac,pc,mid)
riiprime.dot(entACPC)


crossHairNode = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLCrosshairNode')
crossHairRAS = np.array(crossHairNode.GetCrosshairRAS())
originCoordsCross=applyTransformToPoints(slicer.util.getNode('frame_rotation'), crossHairRAS, reverse=True)


entry=[0]*3
getNode('RSTN_line').GetNthControlPointPositionWorld(1, entry)
entry=np.array(entry)
target=[0]*3
getNode('RSTN_line').GetNthControlPointPositionWorld(0, target)
target=np.array(target)

rot_ACPC=Targets_to_Frame(entry,target)
rxy_matrix=rotation_matrix(rot_ACPC[0],rot_ACPC[1],0)
wmat=np.matmul(riiprime,rxy_matrix)
rxy_matrix.dot(crossHairRAS)

riiprime=getFrameRotation(ac,pc,mid)
riiprime.T.dot(crossHairRAS)-fc+100


def norm_vec(P1, P2):
	DirVec = [P2[0]-P1[0], P2[1]-P1[1], P2[2]-P1[2]]
	MagVec = np.sqrt([np.square(DirVec[0]) + np.square(DirVec[1]) + np.square(DirVec[2])])
	NormVec = np.array([float(DirVec[0]/MagVec), float(DirVec[1]/MagVec), float(DirVec[2]/MagVec)])
	return NormVec

def mag_vec(P1, P2):
	DirVec = [P2[0]-P1[0], P2[1]-P1[1], P2[2]-P1[2]]
	MagVec = np.sqrt([np.square(DirVec[0]) + np.square(DirVec[1]) + np.square(DirVec[2])])
	return MagVec

def getFrameCenter():
	world=True
	fidNode = slicer.util.getFirstNodeByClassByName('vtkMRMLMarkupsFiducialNode', 'frame_top_bottom')
	a_top = [0] * 3
	c_bot = [0] * 3
	d_top = [0] * 3
	f_bot = [0] * 3
	g_bot = [0] * 3
	i_top = [0] * 3
	for ifid in range(fidNode.GetNumberOfFiducials()):
		if 'A_top' in fidNode.GetNthFiducialLabel(ifid):
			if world:
				fidNode.GetNthControlPointPositionWorld(ifid, a_top)
			else:
				fidNode.GetNthControlPointPosition(ifid, a_top)
		elif 'C_bot' in fidNode.GetNthFiducialLabel(ifid):
			if world:
				fidNode.GetNthControlPointPositionWorld(ifid, c_bot)
			else:
				fidNode.GetNthControlPointPosition(ifid, c_bot)
		elif 'D_top' in fidNode.GetNthFiducialLabel(ifid):
			if world:
				fidNode.GetNthControlPointPositionWorld(ifid, d_top)
			else:
				fidNode.GetNthControlPointPosition(ifid, d_top)
		elif 'F_bot' in fidNode.GetNthFiducialLabel(ifid):
			if world:
				fidNode.GetNthControlPointPositionWorld(ifid, f_bot)
			else:
				fidNode.GetNthControlPointPosition(ifid, f_bot)
		elif 'G_bot' in fidNode.GetNthFiducialLabel(ifid):
			if world:
				fidNode.GetNthControlPointPositionWorld(ifid, g_bot)
			else:
				fidNode.GetNthControlPointPosition(ifid, g_bot)
		elif 'I_top' in fidNode.GetNthFiducialLabel(ifid):
			if world:
				fidNode.GetNthControlPointPositionWorld(ifid, i_top)
			else:
				fidNode.GetNthControlPointPosition(ifid, i_top)
	midB = np.array(a_top) + norm_vec(a_top, c_bot) * (mag_vec(a_top, c_bot) / 2)
	midE = np.array(d_top) + norm_vec(d_top, f_bot) * (mag_vec(d_top, f_bot) / 2)
	midH = np.array(i_top) + norm_vec(i_top, g_bot) * (mag_vec(i_top, g_bot) / 2)
	FC = np.array([(midB[0] + midH[0]) / 2, (midB[1] + midH[1]) / 2, (midB[2] + midH[2]) / 2])
	frameCentNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')
	frameCentNode.SetName('frame_center')
	n = frameCentNode.AddControlPointWorld(vtk.vtkVector3d(FC[0], FC[1], FC[2]))
	frameCentNode.SetNthControlPointLabel(n, 'frame_center')
	frameCentNode.SetNthControlPointLocked(n, True)
	n = frameCentNode.AddControlPointWorld(vtk.vtkVector3d(midB[0], midB[1], midB[2]))
	frameCentNode.SetNthControlPointLabel(n, 'midB')
	frameCentNode.SetNthControlPointLocked(n, True)
	n = frameCentNode.AddControlPointWorld(vtk.vtkVector3d(midE[0], midE[1], midE[2]))
	frameCentNode.SetNthControlPointLabel(n, 'midE')
	frameCentNode.SetNthControlPointLocked(n, True)
	n = frameCentNode.AddControlPointWorld(vtk.vtkVector3d(midH[0], midH[1], midH[2]))
	frameCentNode.SetNthControlPointLabel(n, 'midH')
	frameCentNode.SetNthControlPointLocked(n, True)
	return FC