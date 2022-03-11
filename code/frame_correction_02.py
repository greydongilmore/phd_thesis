def getPlaneIntersectionPoint():
	axialNode = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNodeRed')
	ortho1Node = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNodeYellow')
	ortho2Node = slicer.mrmlScene.GetNodeByID('vtkMRMLSliceNodeGreen')
	axialSliceToRas = axialNode.GetSliceToRAS()
	n1 = [axialSliceToRas.GetElement(0,2),axialSliceToRas.GetElement(1,2),axialSliceToRas.GetElement(2,2)]
	x1 = [axialSliceToRas.GetElement(0,3),axialSliceToRas.GetElement(1,3),axialSliceToRas.GetElement(2,3)]
	ortho1SliceToRas = ortho1Node.GetSliceToRAS()
	n2 = [ortho1SliceToRas.GetElement(0,2),ortho1SliceToRas.GetElement(1,2),ortho1SliceToRas.GetElement(2,2)]
	x2 = [ortho1SliceToRas.GetElement(0,3),ortho1SliceToRas.GetElement(1,3),ortho1SliceToRas.GetElement(2,3)]
	ortho2SliceToRas = ortho2Node.GetSliceToRAS()
	n3 = [ortho2SliceToRas.GetElement(0,2),ortho2SliceToRas.GetElement(1,2),ortho2SliceToRas.GetElement(2,2)]
	x3 = [ortho2SliceToRas.GetElement(0,3),ortho2SliceToRas.GetElement(1,3),ortho2SliceToRas.GetElement(2,3)]
	# Computed intersection point of all planes
	x = [0,0,0]    
	n2_xp_n3 = [0,0,0]
	x1_dp_n1 = vtk.vtkMath.Dot(x1,n1)
	vtk.vtkMath.Cross(n2,n3,n2_xp_n3)
	vtk.vtkMath.MultiplyScalar(n2_xp_n3, x1_dp_n1)
	vtk.vtkMath.Add(x,n2_xp_n3,x)
	n3_xp_n1 = [0,0,0]
	x2_dp_n2 = vtk.vtkMath.Dot(x2,n2)
	vtk.vtkMath.Cross(n3,n1,n3_xp_n1)
	vtk.vtkMath.MultiplyScalar(n3_xp_n1, x2_dp_n2)
	vtk.vtkMath.Add(x,n3_xp_n1,x)
	n1_xp_n2 = [0,0,0]
	x3_dp_n3 = vtk.vtkMath.Dot(x3,n3)
	vtk.vtkMath.Cross(n1,n2,n1_xp_n2)
	vtk.vtkMath.MultiplyScalar(n1_xp_n2, x3_dp_n3)
	vtk.vtkMath.Add(x,n1_xp_n2,x)
	normalMatrix = vtk.vtkMatrix3x3()
	normalMatrix.SetElement(0,0,n1[0])
	normalMatrix.SetElement(1,0,n1[1])
	normalMatrix.SetElement(2,0,n1[2])
	normalMatrix.SetElement(0,1,n2[0])
	normalMatrix.SetElement(1,1,n2[1])
	normalMatrix.SetElement(2,1,n2[2])
	normalMatrix.SetElement(0,2,n3[0])
	normalMatrix.SetElement(1,2,n3[1])
	normalMatrix.SetElement(2,2,n3[2])
	normalMatrixDeterminant = normalMatrix.Determinant()
	if abs(normalMatrixDeterminant)>0.01:
	  vtk.vtkMath.MultiplyScalar(x, 1/normalMatrixDeterminant)
	else:
	  x = x1
	return x

import numpy as np
def norm_vec(P1, P2):
	DirVec = [P2[0]-P1[0], P2[1]-P1[1], P2[2]-P1[2]]
	MagVec = np.sqrt([np.square(DirVec[0]) + np.square(DirVec[1]) + np.square(DirVec[2])])
	NormVec = np.array([float(DirVec[0]/MagVec), float(DirVec[1]/MagVec), float(DirVec[2]/MagVec)])
	return NormVec

def mag_vec(P1, P2):
	DirVec = [P2[0]-P1[0], P2[1]-P1[1], P2[2]-P1[2]]
	MagVec = np.sqrt([np.square(DirVec[0]) + np.square(DirVec[1]) + np.square(DirVec[2])])
	return MagVec



def CT_to_frame( TP, FC):
	frame_target_X = TP[0]-FC[0] + 100
	frame_target_Y = TP[1]-FC[1] + 100
	frame_target_Z = TP[2]-FC[2] + 100
	return np.array([frame_target_X, frame_target_Y, frame_target_Z])




transformMatrix = vtk.vtkGeneralTransform()
slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(transform, None, transformMatrix)
transformMatrix.TransformPoint(Entry_3)

crossHairRAS=np.array(slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLCrosshairNode').GetCrosshairRAS())
crossHairRAS=np.array(applyTransformToPoints(list(slicer.util.getNodes('*acpc_transform*').values())[0], crossHairRAS, reverse=True))

crossHairRAS=np.array(applyTransformToPoints(list(slicer.util.getNodes('*desc-affine_from-ctFrame*').values())[0], crossHairRAS, reverse=True))
np.array(applyTransformToPoints(list(slicer.util.getNodes('*acpc_transform*').values())[0], crossHairRAS, reverse=True))

mcp=[0]*3
getNode('acpc').GetNthControlPointPosition(2, mcp)
mcp=np.array(mcp)
FC=np.array(applyTransformToPoints(list(slicer.util.getNodes('*desc-affine_from-ctFrame*').values())[0], np.array(FC), reverse=False))

arcToFrameTransform=slicer.vtkMRMLLinearTransformNode()
arcToFrameTransform.SetName("ct_rev")
transformMatrix2 = vtk.vtkGeneralTransform()
slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, transform, transformMatrix2)
arcToFrameTransform.SetMatrixTransformToParent(transformMatrix)
slicer.mrmlScene.AddNode(arcToFrameTransform)

transform=list(slicer.util.getNodes('*desc-affine_from-ctFrame*').values())[0]
transformMatrix = vtk.vtkMatrix4x4()
transform.GetMatrixTransformToParent(transformMatrix)

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


def angleBetweenLineAndSlice(startPoint,endPoint, sliceView):
	lineDirectionVector = (endPoint-startPoint)/np.linalg.norm(endPoint-startPoint)
	sliceToRAS = slicer.mrmlScene.GetNodeByID(sliceView).GetSliceToRAS()
	sliceNormalVector = np.array([sliceToRAS.GetElement(0,2), sliceToRAS.GetElement(1,2), sliceToRAS.GetElement(2,2)])
	angleRad = vtk.vtkMath.AngleBetweenVectors(sliceNormalVector, lineDirectionVector)
	angleDeg = vtk.vtkMath.DegreesFromRadians(angleRad)
	return angleDeg


entryL=[0]*3
getNode('LSTN_line').GetNthControlPointPosition(1, entryL)
entryL=np.array(entryL)
targetL=[0]*3
getNode('LSTN_line').GetNthControlPointPosition(0, targetL)
targetL=np.array(targetL)

sliceView='vtkMRMLSliceNodeRed'
angleBetweenLineAndSlice(entryL, targetL, sliceView)


def applyTransformToPoints( transform, points, reverse=False):
	transformMatrix = vtk.vtkGeneralTransform()
	if reverse:
		slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, transform, transformMatrix)
	else:
		slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(transform, None, transformMatrix)
	finalPoints = transformMatrix.TransformPoint(points)
	return np.array(finalPoints)


def getSliceFrameFids(targetZ):
	fidNode = slicer.util.getFirstNodeByClassByName('vtkMRMLMarkupsFiducialNode', 'frame_fids')
	labels=['P1','P2','P3','P4','P5','P6','P7','P8','P9']
	points = []
	cnt=0
	for ilabel in labels:
		frameFids=[]
		for i in range(fidNode.GetNumberOfFiducials()):
			if ilabel in fidNode.GetNthFiducialLabel(i):
				points_tmp=[0]*3
				fidNode.GetNthControlPointPositionWorld(i, points_tmp)
				frameFids.append(list(points_tmp))
		frameFids=np.vstack(frameFids)
		zSlice=(np.abs([np.round(x[2],1) for x in frameFids] - targetZ)).argmin()
		points.append(frameFids[zSlice,:])
	points=np.vstack(points)
	return points

sliceFrameFids=getSliceFrameFids(np.round(fc[2],1))
pointA=sliceFrameFids[8,:]
pointB=sliceFrameFids[7,:]
pointC=sliceFrameFids[6,:]
pointD=sliceFrameFids[5,:]
pointE=sliceFrameFids[4,:]
pointF=sliceFrameFids[3,:]
pointG=sliceFrameFids[2,:]
pointH=sliceFrameFids[1,:]
pointI=sliceFrameFids[0,:]

barAtop=np.array([5,40,40])
barCbot=np.array([5,160,160])
barDtop=np.array([40,215,40])
barFbot=np.array([160,215,160])
barGbot=np.array([195,160,160])
barItop=np.array([195,40,40])

barAmid=np.array([5,40,100])
barCmid=np.array([5,160,100])
barDmid=np.array([40,215,100])
barFmid=np.array([160,215,100])
barGmid=np.array([195,160,100])
barImid=np.array([195,40,100])




# correction for tile angle
X0 = ((pointC[0] + pointG[0])/2 + (pointA[0] + pointI[0])/2 + (pointC[0] + pointI[0])/2 + (pointA[0] + pointG[0])/2)/4
Y0 = ((pointA[1] + pointC[1])/2 + (pointG[1] + pointI[1])/2 + (pointA[1] + pointG[1])/2 + (pointC[1] + pointI[1])/2)/4
CB=((pointC[0]-pointB[0])**2 + (pointC[1]-pointB[1])**2)**.5
HI=((pointH[0]-pointI[0])**2 + (pointH[1]-pointI[1])**2)**.5
FE=((pointF[0]-pointE[0])**2 + (pointF[1]-pointE[1])**2)**.5
Zs=((CB+HI)/2*math.radians(np.tan(45)))*60

C=CB/(math.radians(np.tan(45)) + 60)
F=FE/(math.radians(np.tan(45)) + 60)
I=HI/(math.radians(np.tan(45)) + 60)

alpha=np.arctan((C-I)/(2*X0))
beta=np.arctan(((2*F) - C - I)/(2*Y0))

Xt = 87*np.cos(alpha)
Yt = 94.15*np.cos(beta)

AB=((pointB[0]-pointA[0])**2 + (pointB[1]-pointA[1])**2)**.5
AC=((pointC[0]-pointA[0])**2 + (pointC[1]-pointA[1])**2)**.5
f1=AB/AC
frameB=(f1*barAtop) + ((1-f1)*barCbot)

DE=((pointE[0]-pointD[0])**2 + (pointE[1]-pointD[1])**2)**.5
DF=((pointF[0]-pointD[0])**2 + (pointF[1]-pointD[1])**2)**.5
f2=DE/DF
frameE=(f2*barDtop) + ((1-f2)*barFbot)

GH=((pointH[0]-pointG[0])**2 + (pointH[1]-pointG[1])**2)**.5
GI=((pointI[0]-pointG[0])**2 + (pointI[1]-pointG[1])**2)**.5
f3=GH/GI
frameH=(f3*barItop) + ((1-f3)*barGbot)

pointB=ras_to_ijk(pointB.tolist(), getNode('rsub-P115_acq-Frame_ct'))
pointE=ras_to_ijk(pointE.tolist(), getNode('rsub-P115_acq-Frame_ct'))
pointH=ras_to_ijk(pointH.tolist(), getNode('rsub-P115_acq-Frame_ct'))

S=np.vstack((np.hstack((pointB, mcp_ct[0])), np.hstack((pointE,mcp_ct[1])), np.hstack((pointH,mcp_ct[2])), np.array(([0,0,0,1]))))
F=np.vstack((np.hstack((np.array(frameB), fc[0])), np.hstack((np.array(frameE),fc[1])), np.hstack((np.array(frameH),fc[2])), np.array(([0,0,0,1]))))

M=np.matmul(np.linalg.inv(S),F)

test=entAP
(np.linalg.inv(M).dot(np.hstack((entAP, 1)))[:3]*np.array([-1,1,-1]))+fc_frame

entry=np.array([134.7,138.2,26.6])
np.linalg.inv(M).dot(entry)


centerOfRotationMarkupsNode=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')
centerOfRotationMarkupsNode.SetAndObserveTransformNodeID(list(slicer.util.getNodes('*desc-affine_from-ctFrame*').values())[0].GetID())
n = centerOfRotationMarkupsNode.AddControlPointWorld(vtk.vtkVector3d(midA1[0],midA1[1],midA1[2]))
centerOfRotationMarkupsNode.SetNthControlPointLabel(n, 'midA1')
centerOfRotationMarkupsNode.SetNthControlPointLocked(n, True)
n = centerOfRotationMarkupsNode.AddControlPointWorld(vtk.vtkVector3d(midA2[0],midA2[1],midA2[2]))
centerOfRotationMarkupsNode.SetNthControlPointLabel(n, 'midA2')
centerOfRotationMarkupsNode.SetNthControlPointLocked(n, True)
n = centerOfRotationMarkupsNode.AddControlPointWorld(vtk.vtkVector3d(midA3[0],midA3[1],midA3[2]))
centerOfRotationMarkupsNode.SetNthControlPointLabel(n, 'midA1')
centerOfRotationMarkupsNode.SetNthControlPointLocked(n, True)



def getObliqueCoords(pointA, pointB, pointC):
	f=((pointB[0]-pointA[0])**2 + (pointB[1]-pointA[1])**2)**.5/((pointC[0]-pointA[0])**2 + (pointC[1]-pointA[1])**2)**.5
	B3=(f*pointC) + ((1-f)*pointA)
	return B3

B3=getObliqueCoords(sliceFrameFids[8,:], sliceFrameFids[7,:], sliceFrameFids[6,:])
B3_frame=CT_to_frame_coords(B3, FC)

B2=getObliqueCoords(sliceFrameFids[5,:], sliceFrameFids[4,:], sliceFrameFids[3,:])
B2_frame=CT_to_frame(B2, FC)

B1=getObliqueCoords(sliceFrameFids[2,:], sliceFrameFids[1,:], sliceFrameFids[0,:])
B1_frame=CT_to_frame(B1, FC)

CT_to_frame(np.array([1.012,36.349,27.041]), FC)

M = np.matmul(, np.vstack((np.array(B1), np.array(B2), np.array(B3))))

np.round(np.matmul(np.linalg.inv(M),np.array([1.012,36.349,1])),2)
(mag_vec(sliceFrameFids[7,:],sliceFrameFids[8,:]) + mag_vec(sliceFrameFids[0,:],sliceFrameFids[1,:]))/2+40


originCoordsMCP=[0]*3
getNode('acpc').GetNthControlPointPosition(2, originCoordsMCP)
newCoordsCTtemp=applyTransformToPoints(list(slicer.util.getNodes('*desc-affine_from-ctFrame*').values())[0], np.array(originCoordsMCP)+np.array([29.47,52.83,73.29]), reverse=True)
newCoordsCT=applyTransformToPoints(list(slicer.util.getNodes('*acpc_transform*').values())[0], newCoordsCTtemp, reverse=False)
newCoordsCT
fcCoords=[0]*3
getNode('frame_center').GetNthControlPointPositionWorld(0, fcCoords)

newCoordsCT=applyTransformToPoints(list(slicer.util.getNodes('*sub-P069_desc-rigid_from-ctFrame_to-T1w_xfm_reverse*').values())[0], ent)



fidList=['midA1','midA2','midA3']

lineNode = slicer.util.getNode('acpc')
lineStartPos = np.zeros(3)
lineEndPos = np.zeros(3)
for i in range(lineNode.GetNumberOfControlPoints()):
	if 'ac' in lineNode.GetNthControlPointLabel(i):
		lineNode.GetNthControlPointPositionWorld(i, lineStartPos)
	elif 'pc' in lineNode.GetNthControlPointLabel(i):
		lineNode.GetNthControlPointPositionWorld(i, lineEndPos)
lineDirectionVector = (lineEndPos-lineStartPos)/np.linalg.norm(lineEndPos-lineStartPos)
fidNode = slicer.util.getNode('MarkupsFiducial_20')
nOfFiduciallPoints = 0
for i in range(fidNode.GetNumberOfControlPoints()):
	if any(x in fidNode.GetNthControlPointLabel(i) for x in fidList):
		nOfFiduciallPoints +=1
points = np.zeros([3, nOfFiduciallPoints])
cnt=0
for i in range(fidNode.GetNumberOfControlPoints()):
	if any(x in fidNode.GetNthControlPointLabel(i) for x in fidList):
		fidNode.GetNthControlPointPositionWorld(i, points[:,cnt])
		cnt+=1
planePosition = points.mean(axis=1)
planeNormal = np.cross(points[:,1] - points[:,0], points[:,2] - points[:,0])
angleRad = vtk.vtkMath.AngleBetweenVectors(planeNormal, lineDirectionVector)
angleDeg = vtk.vtkMath.DegreesFromRadians(angleRad)

fidName='MarkupsFiducial_20'
fidList=['midA1','midA2','midA3']
sliceView='vtkMRMLSliceNodeRed'
def angleBetweenFiducialPlaneAndSlice(fidName, fidList, sliceView):
	fidNode = slicer.util.getFirstNodeByClassByName('vtkMRMLMarkupsFiducialNode', fidName)
	nOfFiduciallPoints = 0
	for i in range(fidNode.GetNumberOfFiducials()):
		if any(x in fidNode.GetNthFiducialLabel(i) for x in fidList):
			nOfFiduciallPoints +=1
	points = np.zeros([3, nOfFiduciallPoints])
	cnt=0
	for i in range(fidNode.GetNumberOfFiducials()):
		if any(x in fidNode.GetNthFiducialLabel(i) for x in fidList):
			fidNode.GetNthControlPointPositionWorld(i, points[:,cnt])
			cnt+=1
	planePosition = points.mean(axis=1)
	planeNormal = np.cross(points[:,1] - points[:,0], points[:,2] - points[:,0])
	sliceToRAS = slicer.mrmlScene.GetNodeByID(sliceView).GetSliceToRAS()
	sliceNormalVector = np.array([sliceToRAS.GetElement(0,2), sliceToRAS.GetElement(1,2), sliceToRAS.GetElement(2,2)])
	angleRad = vtk.vtkMath.AngleBetweenVectors(planeNormal, sliceNormalVector)
	angleDeg = vtk.vtkMath.DegreesFromRadians(angleRad)
	return angleDeg




mcp=[0]*3
getNode('acpc').GetNthControlPointPosition(2, mcp)
mcp=np.array(mcp)
ac=[0]*3
getNode('acpc').GetNthControlPointPosition(0, ac)
ac=np.array(ac)
pc=[0]*3
getNode('acpc').GetNthControlPointPosition(1, pc)
pc=np.array(pc)
mid=[0]*3
getNode('midline').GetNthControlPointPosition(0, mid)
mid=np.array(mid)
fc=[0]*3
getNode('frame_center').GetNthControlPointPosition(0, fc)
fc=np.array(fc)

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

ent=np.array([32.95,41.49,67.38])

entryL=[0]*3
getNode('LSTN_line').GetNthControlPointPosition(1, entryL)
entryL=np.array(entryL)
targetL=[0]*3
getNode('LSTN_line').GetNthControlPointPosition(0, targetL)
targetL=np.array(targetL)
entryR=[0]*3
getNode('RSTN_line').GetNthControlPointPosition(1, entryR)
entryR=np.array(entryR)
targetR=[0]*3
getNode('RSTN_line').GetNthControlPointPosition(0, targetR)
targetR=np.array(targetR)


target=np.array([-10.750,-13.04,-20.330])

def frame_coordinates(ac, pc, mid):
	MCP_frame = (ac+pc)/2
	IC = ac-pc
	len_IC = np.sqrt((IC[0])**2+(IC[1])**2+(IC[2])**2)
	e_y = np.array([IC[0]/len_IC, IC[1]/len_IC, IC[2]/len_IC])
	w = np.array(mid)-np.array(MCP_frame)
	if float(np.diff((MCP_frame[2], mid[2]))) > 0:
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

def Targets_to_Frame(Xt, Xe):
	################################################
	# Purpose: To convert a formal description of a cylinder, the target point, and the point of entry into coordinates for a cylindrically shaped stereotactic head frame.
	# Inputs:
	#      Xt: the target point
	#      Xe: point of entry into the patient
	# Outputs:
	#     arc: The arc angle neccessary for a trajectory to go through point Xt and Xe
	#  collar: The collar angle neccessary for a trajectory to go through point Xt and Xe
	################################################
	# unit vector from target to entry point
	Xt = np.array(Xt)
	Xe = np.array(Xe)
	Xr = np.array(Xe-Xt)
	dist = np.linalg.norm(Xr) #The distance between the points.
	phi=np.array([0.0,0.0])#arc and collar
	phi[0]=np.arccos(Xr[0]/dist)
	if Xr[1]!=0:
		phi[1]=np.arctan(Xr[2]/Xr[1])
	else:
		phi[1]=np.pi/2.0
	if phi[1]<0:
		phi[1]=np.pi+phi[1]
	return [math.degrees(phi[0]),math.degrees(phi[1])]
	
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
	if phi[1]>0:
		phi[1]=np.pi+phi[1]
	return [math.degrees(phi[0]),math.degrees(phi[1])]

def CT_to_frame_coords(TP, FC=None):
	frame_target =[]
	tp_sign_x = np.sign(TP[0])*-1
	tp_sign_y = TP[1]
	tp_sign_z = np.sign(TP[2])*-1
	if tp_sign_x > 0:
		frame_target.append(100+abs(TP[0]))
	else:
		frame_target.append(100-abs(TP[0]))
	if tp_sign_y > 0:
		frame_target.append(100+abs(TP[1]))
	else:
		frame_target.append(100-abs(TP[1]))
	if tp_sign_z > 0:
		frame_target.append(100+abs(TP[2]))
	else:
		frame_target.append(100-abs(TP[2]))
	return frame_target

def getFrameRotation(ac, pc, mid):
	pmprime = (ac+pc)/2
	vec1=ac-pmprime
	vec2=mid-pmprime
	vec1Mag=np.sqrt(vec1[0]**2 + vec1[1]**2 + vec1[2]**2)
	vec2Mag=np.sqrt(vec2[0]**2 + vec2[1]**2 + vec2[2]**2)
	vec1Unit=vec1/vec1Mag
	vec2Unit=vec2/vec2Mag
	# Anteroposterior axis
	yihatprime = vec1Unit
	# Lateral axis
	if pmprime[2] > mid[2]:
		xihatprime=-1*np.cross(vec1Unit,vec2Unit)
	else:
		xihatprime=np.cross(vec1Unit,vec2Unit)
	xAxisMag=(xihatprime[0]**2 + xihatprime[1]**2 + xihatprime[2]**2)**.5
	# Rostrocaudal axis
	zAxis = np.cross(yihatprime,xihatprime)
	xihatprime=xihatprime/xAxisMag
	zAxisMag=(zAxis[0]**2 + zAxis[1]**2 + zAxis[2]**2)**.5
	zihatprime=zAxis/zAxisMag
	#standard basis
	xihat=np.array([1,0,0])
	yihat=np.array([0,1,0])
	zihat=np.array([0,0,1])
	# Rotation matrix
	riiprime = np.vstack([np.array([xihatprime.dot(xihat), xihatprime.dot(yihat), xihatprime.dot(zihat)]),
						np.array([yihatprime.dot(xihat), yihatprime.dot(yihat), yihatprime.dot(zihat)]),
						np.array([zihatprime.dot(xihat), zihatprime.dot(yihat), zihatprime.dot(zihat)])
						])
	return riiprime


newcrwP=entAP2-mcp2
computeframetoMidACPC=riiprime2.dot(newcrwP)
currentFramecalc=riiprime2.T.dot(computeframetoMidACPC)
currentframecalcPM=currentFramecalc+mcp2
currentframecalcPM
xyzpmdelta=currentframecalcPM-mcp2
computeanatomicfromframe=riiprime.T.dot(xyzpmdelta)


#sub-P115
entAP=np.array([-34.13,44.9,71.22])
tarAP=np.array([-12.36,-0.81,-3.18])
entAP=np.array([29.47,52.83,73.29])
tarAP=np.array([12.71,-1.7,-3.92])

#sub-P150
entAP=np.array([43.48,25.66,68.93])
tarAP=np.array([11.5,-2.5,-4.62])

#sub-P155
entAP=np.array([-34.43,53.91,58.96])
tarAP=np.array([-11.54,-2.14,-3.93])

#sub-P175
entAP=np.array([46.32,28.88,68.7])
tarAP=np.array([10.49,-7.46,-4.22])

ac=applyTransformToPoints(getNode('acpc_transform'), ac)


ac=applyTransformToPoints(list(slicer.util.getNodes('*acpc_transform*').values())[0], ac, reverse=True)
pc=applyTransformToPoints(list(slicer.util.getNodes('*acpc_transform*').values())[0], pc, reverse=True)
mcp=applyTransformToPoints(list(slicer.util.getNodes('*acpc_transform*').values())[0], mcp, reverse=True)
mid=applyTransformToPoints(list(slicer.util.getNodes('*acpc_transform*').values())[0], mid, reverse=True)
fc=applyTransformToPoints(list(slicer.util.getNodes('*acpc_transform*').values())[0], fc, reverse=True)

ac=applyTransformToPoints(list(slicer.util.getNodes('*acpc_transform*').values())[0], ac, reverse=True)
pc=applyTransformToPoints(list(slicer.util.getNodes('*acpc_transform*').values())[0], pc, reverse=True)
mcp=applyTransformToPoints(list(slicer.util.getNodes('*acpc_transform*').values())[0], mcp, reverse=True)
mid=applyTransformToPoints(list(slicer.util.getNodes('*acpc_transform*').values())[0], mid, reverse=True)
fc=applyTransformToPoints(list(slicer.util.getNodes('*acpc_transform*').values())[0], fc, reverse=True)

ac=applyTransformToPoints(slicer.util.getNode('from-fiducials_to-localizer_transform'), ac, reverse=True)
pc=applyTransformToPoints(slicer.util.getNode('from-fiducials_to-localizer_transform'), pc, reverse=True)
mcp=applyTransformToPoints(slicer.util.getNode('from-fiducials_to-localizer_transform'), mcp, reverse=True)
mid=applyTransformToPoints(slicer.util.getNode('from-fiducials_to-localizer_transform'), mid, reverse=True)
fc=applyTransformToPoints(slicer.util.getNode('from-fiducials_to-localizer_transform'), fc, reverse=True)

ac_frame = np.array(CT_to_frame_coords(ac-fc))
pc_frame = np.array(CT_to_frame_coords(pc-fc))
mid_frame = np.array(CT_to_frame_coords(mid-fc))
fc_frame = np.array(CT_to_frame_coords(fc))
mcp_frame = (ac_frame+pc_frame)/2

def ras_to_ijk(point, node):
	rasToijk = vtk.vtkMatrix4x4()
	node.GetRASToIJKMatrix(rasToijk)
	position_ras = point + [1]
	point_Ijk = rasToijk.MultiplyDoublePoint(position_ras)
	return np.array(point_Ijk[:3])

ac_in=[0]*3
getNode('acpc').GetNthFiducialPosition(0, ac_in)
ac_ct=ras_to_ijk(ac_in, getNode('rsub-P115_acq-Frame_ct'))
pc_in=[0]*3
getNode('acpc').GetNthFiducialPosition(1, pc_in)
pc_ct=ras_to_ijk(pc_in, getNode('rsub-P115_acq-Frame_ct'))
mid_in=[0]*3
getNode('midline').GetNthFiducialPosition(0, mid_in)
mid_ct=ras_to_ijk(mid_in, getNode('rsub-P115_acq-Frame_ct'))
fc_in=[0]*3
getNode('frame_center').GetNthFiducialPosition(0, fc_in)
fc_ct=ras_to_ijk(fc_in, getNode('rsub-P115_acq-Frame_ct'))
mcp_ct = (ac_ct+pc_ct)/2

ent_ct=ras_to_ijk((mcp+entAP).tolist(), getNode('rsub-P115_acq-Frame_ct'))
newcrwP=entAP-mcp
riiprime=getFrameRotation(ac_ct,pc_ct,mid_ct)
computeframetoMidACPC=riiprime.dot(entAP-mcp)*np.array([-1,1,-1])+mcp_frame


currentframecalcPM=riiprime.T.dot(ent_ct)*np.array([-1,1,-1])+mcp_frame


ac_ct=np.array([-3.5,238.3,-9.9])
pc_ct=np.array([-3.3,213.9,-18.7])
mid_ct=np.array([-4.8,244.8,33.9])
mcp_ct = (ac_ct+pc_ct)/2

newcrwP=np.array([-16.5,220.2,-13.8])-mcp_ct

riiprime=getFrameRotation(ac,pc,mid)
(riiprime.dot(entAP)*np.array([-1,1,-1]))+mcp_frame

currentFramecalc=riiprime_frame.dot(entAP+mcp)
currentframecalcPM=((currentFramecalc)*np.array([-1,1,-1]))+fc_frame
currentframecalcPM
xyzpmdelta=(currentframecalcPM-mcp_frame-(mcp_frame-fc_frame))*np.array([-1,1,-1])
computeanatomicfromframe=riiprime.dot(xyzpmdelta+fc)
computeanatomicfromframe



#get frame settings
riiprime = getFrameRotation(ac, pc, mid)
(riiprime.dot(entryL)*np.array([-1,1,-1]))+mcp_frame

frame_entry=np.array(CT_to_frame_coords(riiprime.T.dot(entryR)))
frame_target=np.array(CT_to_frame_coords(riiprime.T.dot(targetR)))


CT_to_frame_coords(targetR)
CT_to_frame_coords(entryR)
Targets_to_Frame(riiprime.dot(targetR-fc), riiprime.dot(entryR-fc))

arc, ring = Targets_to_Frame(riiprime.T.dot(targetR), riiprime.T.dot(entryR))


riiprime.T.dot(entAP)*np.array([-1,1,-1])+mcp_frame+(fc_frame-mcp_frame)

riiprime.T.dot(entAP)-mcp_frame*np.array([-1,1,-1])

CT_to_frame_coords(riiprime.T.dot(entAP+(mcp)))

np.matmul(riiprime.T, entAP)
riiprime.dot(tarAP)-(mcp_frame-fc_frame)

entAP=np.array([46.32,28.88,68.7])
tarAP=np.array([10.49,-7.46,-4.22])
tarFrame=riiprime.dot(entAP)
if tarFrame[0] < 0:
	tarFrame=np.array([abs(tarFrame[0]), tarFrame[1], tarFrame[2]])
tarFrame+mcp_frame

tar_frame = np.array(CT_to_frame_coords(tarAP))
tar_frame+mcp_frame
riiprime.dot(tarAP)+mcp_frame
CT_to_frame_coords(riiprime.dot(entAP))
np.array(CT_to_frame_coords(riiprime.dot(entAP), fc))
ac=[0]*3
getNode('acpc').GetNthControlPointPositionWorld(0, ac)
ac=np.array(ac)
ac=applyTransformToPoints(list(slicer.util.getNodes('*acpc_transform*').values())[0], ac, reverse=True)

pc=[0]*3
getNode('acpc').GetNthControlPointPositionWorld(1, pc)
pc=np.array(pc)
pc=applyTransformToPoints(list(slicer.util.getNodes('*acpc_transform*').values())[0], pc, reverse=True)

mcp=[0]*3
getNode('acpc').GetNthControlPointPositionWorld(2, mcp)
mcp=np.array(mcp)
mcp=applyTransformToPoints(list(slicer.util.getNodes('*acpc_transform*').values())[0], mcp, reverse=True)

mid=[0]*3
getNode('midline').GetNthControlPointPositionWorld(1, mid)
mid=np.array(mid)
mid=applyTransformToPoints(list(slicer.util.getNodes('*acpc_transform*').values())[0], mid, reverse=True)

fc=[0]*3
getNode('frame_center').GetNthControlPointPositionWorld(0, fc)
fc=np.array(fc)
fc=applyTransformToPoints(list(slicer.util.getNodes('*acpc_transform*').values())[0], fc, reverse=True)


ac_frame = np.array(CT_to_frame_coords(ac, fc))
pc_frame = np.array(CT_to_frame_coords(pc, fc))
mcp_frame = np.array([(ac_frame[0]+pc_frame[0])/2, (ac_frame[1]+pc_frame[1])/2,(ac_frame[2]+pc_frame[2])/2])

riiprime=getframunitvec(ac,pc,mid,fc)
riiprime.dot(entAP)+mcp_frame

CT_to_frame_coords(riiprime.dot(entAP),fc)

frame_coordinates(ac,pc,mid).dot(entAP)+mcp_frame
riiprime.dot(tarAP)+mcp_frame


ent=np.array([-47.04,20.79,72.67])
entAP=mcp+ent
entF=np.array([152.7,118.9,34.3])
mcp = (np.array(ac)-np.array(pc))/2
mcpFramepm=mcpFrame-mcp

frame_to_CT_coords(mcpFrame,fc)

riiprimeCT=getframunitvec(ac, pc, mid)

ac_ct=CT_to_frame_coords(ac,fc)
pc_ct=CT_to_frame_coords(pc,fc)
mid_ct=CT_to_frame_coords(mid,fc)

riiprimeCT=getframunitvec(ac, pc, mid)

S=np.vstack((ac, pc, mid))
F=np.vstack((ac_ct, pc_ct, mid_ct))

M=np.matmul(np.linalg.inv(F),S)

np.matmul(M,entAP)

riiprimeFrame=getframunitvec(ac_ct, pc_ct, mid_ct)

riiprime=np.matmul(riiprimeCT,riiprimeFrame)


currentFramecalc=np.matmul(np.linalg.inv(riiprime), mcpFrame.T)
newFrame=



inverse CT: (harden?)
- imaging except frame CT

acpc_transform:
-ac,pc,mcp,mids,entry,target


getNode('sub-P124_desc-rigid_from-fiducials_to-localizer_xfm')

#sub-P175
tarAP=np.array([-10.313,-2.158,-1.001])
entAP=np.array([-38.405,32.107,64.702])

entAPt=applyTransformToPoints(getNode('sub-P124_desc-rigid_from-fiducials_to-localizer_xfm'), entAP, reverse=False)
entAPt

pc=applyTransformToPoints(list(slicer.util.getNodes('*acpc_transform*').values())[0], pc, reverse=True)
mcp=applyTransformToPoints(list(slicer.util.getNodes('*acpc_transform*').values())[0], mcp, reverse=True)

def calculate_angles(Entry, Target):
	dirVec = Entry-Target
	alpha = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsAngleNode")
	alpha.SetName('alpha')
	alpha.SetAngleMeasurementMode( 2 )
	alpha.SetDisplayVisibility(False)
	pointA = vtk.vtkVector3d( 0,100, 0)
	pointB = vtk.vtkVector3d(0, 0, 0)
	pointC = vtk.vtkVector3d(0 ,dirVec[1],dirVec[2])
	alpha.AddControlPoint( pointA )
	alpha.AddControlPoint( pointB )
	alpha.AddControlPoint( pointC )
	angle_alpha = alpha.GetAngleDegrees()
	beta = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsAngleNode")
	beta.SetName('beta')
	beta.SetAngleMeasurementMode(0)
	beta.SetDisplayVisibility(False)
	if dirVec[0] >= 0: 
		pointA = vtk.vtkVector3d(-100, 0, 0)
	else:
		pointA = vtk.vtkVector3d(100, 0, 0)
	pointB = vtk.vtkVector3d(0, 0, 0)
	pointC = vtk.vtkVector3d( dirVec[0],dirVec[1], dirVec[2])
	beta.AddControlPoint( pointA )
	beta.AddControlPoint(pointB)
	beta.AddControlPoint( pointC )
	angle_beta = beta.GetAngleDegrees()
	return  angle_alpha , angle_beta

calculate_angles(entry,target)

def calcula_angulos(Entry, Target):
	x = Entry[0]-Target[0]
	y = Entry[1]-Target[1]
	z = Entry[2]-Target[2]
	
	alfa = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsAngleNode")
	alfa.SetName('Alfa')
	alfa.SetAngleMeasurementMode(2)
	alfa.SetDisplayVisibility(False)
	puntoA = vtk.vtkVector3d(0, 100, 0)
	puntoB = vtk.vtkVector3d(0, 0, 0)
	puntoC = vtk.vtkVector3d(0, y, z)
	alfa.AddControlPoint(puntoA)
	alfa.AddControlPoint(puntoB)
	alfa.AddControlPoint(puntoC)
	angulo_Alfa = alfa.GetAngleDegrees()

	beta = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsAngleNode")
	beta.SetName('Beta')
	beta.SetAngleMeasurementMode(0)
	beta.SetDisplayVisibility(False)
	if x >= 0:  
		puntoA = vtk.vtkVector3d(100, 0, 0)
	else:
		puntoA = vtk.vtkVector3d(-100, 0, 0)
	puntoB = vtk.vtkVector3d(0, 0, 0)
	puntoC = vtk.vtkVector3d(x, y, z)
	beta.AddControlPoint(puntoA)
	beta.AddControlPoint(puntoB)
	beta.AddControlPoint(puntoC)
	
	angulo_Beta = beta.GetAngleDegrees()
	print(angulo_Alfa,angulo_Beta)
	return angulo_Alfa, angulo_Beta


def applyTransformToPoints( transform, points, reverse=False):
	transformMatrix = vtk.vtkGeneralTransform()
	if reverse:
		slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, transform, transformMatrix)
	else:
		slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(transform, None, transformMatrix)
	finalPoints = transformMatrix.TransformPoint(points)
	return np.array(finalPoints)

crossHairRAS=np.array(slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLCrosshairNode').GetCrosshairRAS())

entry=[0]*3
getNode('RSTN_line').GetNthControlPointPositionWorld(0, entry)
entry=np.array(entry)
target=[0]*3
getNode('RSTN_line').GetNthControlPointPositionWorld(1, target)
target=np.array(target)
calcula_angulos(entry,target)


ent=np.array([-40.02,43.45,56.66])

ac=[0]*3
getNode('acpc').GetNthControlPointPositionWorld(0, ac)
ac=np.array(ac)
pc=[0]*3
getNode('acpc').GetNthControlPointPositionWorld(1, pc)
pc=np.array(pc)
mcp=[0]*3
getNode('acpc').GetNthControlPointPositionWorld(2, mcp)
mcp=np.array(mcp)


crossHairRAS=np.array(slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLCrosshairNode').GetCrosshairRAS())

slicer.util.getNode('sub-P060_T1w')

acT1=applyTransformToPoints(slicer.util.getNode('sub-P060_desc-rigid_from-ctFrame_to-T1w_xfm_reverse'), ac, reverse=True)
pcT1=applyTransformToPoints(slicer.util.getNode('sub-P060_desc-rigid_from-ctFrame_to-T1w_xfm_reverse'), pc, reverse=True)
mcpT1=(acT1+pcT1)/2

mag_vec(acT1,pcT1)

mag_vec(ac,pc)



coordsToFrame = np.array([
	[ 1, 0, 0,coordsACPC[0]],
	[ 0, 1, 0,coordsACPC[1]],
	[ 0, 0, 1,coordsACPC[2]],
	[ 0, 0, 0,   1]
])
np.dot(coordsToFrame, np.append(mcp,1))[:3]


coordsRAS=applyTransformToPoints(slicer.util.getNode('frame_rotation'), crossHairRAS, reverse=False)
ACPCToRAS = np.array([
	[ 1, 0, 0, mcp[0]],
	[ 0, 1, 0, -mcp[1]],
	[ 0, 0, 1, mcp[2]],
	[ 0, 0, 0,   1]
])
coordsRAS=np.dot(ACPCToRAS, np.append(coordsRAS,1))[:3]
coordsRAS
applyTransformToPoints(slicer.util.getNode('frame_rotation'), coordsRAS, reverse=False)
frameToRAS = np.array([
	[-1, 0, 0, 100],
	[ 0, 1, 0, 100],
	[ 0, 0,-1, 100],
	[ 0, 0, 0,   1]
])
coordsFrame=frameToRAS.dot(np.append(coordsRAS,1))[:3]-fc




arcAngle=72.2
ringAngle=71
dist=85.1
target=np.array([86.5,103.0,94])

def rotateTrajectory(target,arcAngle,ringAngle,dist):
	x = np.sin(math.radians(arcAngle))*np.cos(math.radians(ringAngle))
	z = np.sin(math.radians(arcAngle))*np.sin(math.radians(arcAngle))
	y = np.cos(math.radians(arcAngle))
	new_point=np.dot(frameToRAS, np.append(target,1))[:3]+np.array([x,y,z])*dist
	return new_point

rotateTrajectory(target,arcAngle,ringAngle,dist)

arcAngle=72.2
ringAngle=71.00
initDirection = [0, 1, 0]
ringDirection = [1, 0, 0]
arcDirection =  [0, -np.sin(np.deg2rad(ringAngle)), np.cos(np.deg2rad(ringAngle))]

# Create vtk Transform
vtkTransform = vtk.vtkTransform()
vtkTransform.Translate(np.dot(frameToRAS, np.append(np.array([86.5,103.0,94]),1))[:3])
vtkTransform.RotateWXYZ(arcAngle, arcDirection[0], arcDirection[1], arcDirection[2])
vtkTransform.RotateWXYZ(ringAngle, ringDirection[0], ringDirection[1], ringDirection[2])
vtkTransform.RotateWXYZ(90, initDirection[0], initDirection[1], initDirection[2])

frame=slicer.vtkMRMLLinearTransformNode()
frame.SetName('frame')
slicer.mrmlScene.AddNode(frame)
frame.SetMatrixTransformToParent(vtkTransform.GetMatrix())

ac=[0]*3
getNode('acpc').GetNthControlPointPositionWorld(0, ac)
ac=np.array(ac)
pc=[0]*3
getNode('acpc').GetNthControlPointPositionWorld(1, pc)
pc=np.array(pc)
mcp=[0]*3
getNode('acpc').GetNthControlPointPositionWorld(2, mcp)
mcp=np.array(mcp)
mid=[0]*3
getNode('midline').GetNthControlPointPositionWorld(0, mid)
mid=np.array(mid)

fc=[0]*3
getNode('frame_center').GetNthControlPointPositionWorld(0, fc)
fc=np.array(fc)
mcp=[0]*3
getNode('acpc').GetNthControlPointPositionWorld(2, mcp)
mcp=np.array(mcp)


acT=applyTransformToPoints(slicer.util.getNode('sub-P163_desc-rigid_from-ctFrame_to-T1w_xfm_reverse'), ac, reverse=True)
#acT=applyTransformToPoints(slicer.util.getNode('sub-P163_desc-rigid_from-fiducials_to-localizer_xfm'), acT, reverse=True)
pcT=applyTransformToPoints(slicer.util.getNode('sub-P163_desc-rigid_from-ctFrame_to-T1w_xfm_reverse'), pc, reverse=True)
#pcT=applyTransformToPoints(slicer.util.getNode('sub-P163_desc-rigid_from-fiducials_to-localizer_xfm'), pcT, reverse=True)
midT=applyTransformToPoints(slicer.util.getNode('sub-P163_desc-rigid_from-ctFrame_to-T1w_xfm_reverse'), mid, reverse=True)
#midT=applyTransformToPoints(slicer.util.getNode('sub-P163_desc-rigid_from-fiducials_to-localizer_xfm'), midT, reverse=True)
mcpT=(acT+pcT)/2
ent=np.array([-42.06,27.83,75.73])

riiprime=frame_coordinates(acT,pcT,midT)
riiprime.dot(mcp)
riiprime=getFrameRotation(acT,pcT,midT)
riiprime.T.dot(crossHairRAS)


def mid_fiducials():
	u, v, w = [], [], []
	fids={}
	node=list(slicer.util.getNodes('*frame_center*').values())[0]
	for i in range(node.GetNumberOfControlPoints()):
		if 'mid' in node.GetNthControlPointLabel(i):
			p=[0]*3
			node.GetNthControlPointPosition(i,p)
			fids[node.GetNthControlPointLabel(i)]=p
			u.append(p[0])
			v.append(p[1])
			w.append(p[2])
	fraccion_N = [0, 0, 0, 0]
	node=list(slicer.util.getNodes('*desc-topbottom_fids*').values())[0]
	for i in range(node.GetNumberOfControlPoints()):
		if 'mid' in node.GetNthControlPointLabel(i):
			p=[0]*3
			node.GetNthControlPointPosition(i,p)
			fids[node.GetNthControlPointLabel(i)]=p
	# fraccion de z calculado por N-Locators:
	fraccion_N[1] = (fids['midB'][1]-fids['bar_A_mid'][1])/(fids['bar_C_mid'][1]-fids['bar_A_mid'][1])
	fraccion_N[2] = (fids['midE'][0]-fids['bar_D_mid'][0])/(fids['bar_F_mid'][0]-fids['bar_D_mid'][0])
	fraccion_N[3] = (fids['midH'][1]-fids['bar_I_mid'][1])/(fids['bar_G_mid'][1]-fids['bar_I_mid'][1])
	return u, v, w, fraccion_N

barAtop=np.array([5,40,40])
barCbot=np.array([5,160,160])
barDtop=np.array([40,215,40])
barFbot=np.array([160,215,160])
barGbot=np.array([195,160,160])
barItop=np.array([195,40,40])

barAmid=np.array([5,40,100])
barCmid=np.array([5,160,100])
barDmid=np.array([40,215,100])
barFmid=np.array([160,215,100])
barGmid=np.array([195,160,100])
barImid=np.array([195,40,100])

u, v, w, fraccion_N = mid_fiducials()

F = vtk.vtkMatrix3x3()
F.SetElement(0, 0, fraccion_N[1] * barCbot[0] + (1-fraccion_N[1]) * barAtop[0])
F.SetElement(0, 1, fraccion_N[1] * barCbot[1] + (1-fraccion_N[1]) * barAtop[1])
F.SetElement(0, 2, fraccion_N[1] * barCbot[2])
F.SetElement(1, 0, fraccion_N[2] * barDtop[0] + (1-fraccion_N[2]) * barFbot[0])
F.SetElement(1, 1, fraccion_N[2] * barDtop[1] + (1-fraccion_N[2]) * barFbot[1])
F.SetElement(1, 2, fraccion_N[2] * barDtop[2])
F.SetElement(2, 0, fraccion_N[3] * barGbot[0] + (1-fraccion_N[3]) * barItop[0])
F.SetElement(2, 1, fraccion_N[3] * barGbot[1] + (1-fraccion_N[3]) * barItop[1])
F.SetElement(2, 2, fraccion_N[3] * barGbot[2])

S = vtk.vtkMatrix3x3()
S.SetElement(0, 0, u[0])
S.SetElement(0, 1, v[0])
S.SetElement(0, 2, w[0])
S.SetElement(1, 0, u[1])
S.SetElement(1, 1, v[1])
S.SetElement(1, 2, w[1])
S.SetElement(2, 0, u[2])
S.SetElement(2, 1, v[2])
S.SetElement(2, 2, w[2])
S.Invert()

M = vtk.vtkMatrix3x3()
M.Multiply3x3(S, F, M)
M.Transpose()