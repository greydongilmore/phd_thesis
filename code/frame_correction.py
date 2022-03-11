



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

def angleBetweenLineAndSlice(startFid, endFid, markupNode, sliceView):
	lineNode = slicer.util.getNode(markupNode)
	lineStartPos = np.zeros(3)
	lineEndPos = np.zeros(3)
	for i in range(lineNode.GetNumberOfControlPoints()):
		if startFid in lineNode.GetNthControlPointLabel(i):
			lineNode.GetNthControlPointPositionWorld(i, lineStartPos)
		elif endFid in lineNode.GetNthControlPointLabel(i):
			lineNode.GetNthControlPointPositionWorld(i, lineEndPos)
	lineDirectionVector = (lineEndPos-lineStartPos)/np.linalg.norm(lineEndPos-lineStartPos)
	sliceToRAS = slicer.mrmlScene.GetNodeByID(sliceView).GetSliceToRAS()
	sliceNormalVector = np.array([sliceToRAS.GetElement(0,2), sliceToRAS.GetElement(1,2), sliceToRAS.GetElement(2,2)])
	angleRad = vtk.vtkMath.AngleBetweenVectors(sliceNormalVector, lineDirectionVector)
	angleDeg = vtk.vtkMath.DegreesFromRadians(angleRad)
	return angleDeg

def angleBetweenLineAndLine(line01Node, line01Fids, line02Node, line02Fids):
	lineNode = slicer.util.getNode(line01Node)
	lineStartPos = np.zeros(3)
	lineEndPos = np.zeros(3)
	for i in range(lineNode.GetNumberOfControlPoints()):
		if line01Fids[0] in lineNode.GetNthControlPointLabel(i):
			lineNode.GetNthControlPointPosition(i, lineStartPos)
		elif line01Fids[1] in lineNode.GetNthControlPointLabel(i):
			lineNode.GetNthControlPointPosition(i, lineEndPos)
	line1DirectionVector = (lineEndPos-lineStartPos)/np.linalg.norm(lineEndPos-lineStartPos)
	lineNode = slicer.util.getNode(line02Node)
	lineStartPos = np.zeros(3)
	lineEndPos = np.zeros(3)
	for i in range(lineNode.GetNumberOfControlPoints()):
		if line02Fids[0] in lineNode.GetNthControlPointLabel(i):
			lineNode.GetNthControlPointPositionWorld(i, lineStartPos)
		elif line02Fids[1] in lineNode.GetNthControlPointLabel(i):
			lineNode.GetNthControlPointPositionWorld(i, lineEndPos)
	line2DirectionVector = (lineEndPos-lineStartPos)/np.linalg.norm(lineEndPos-lineStartPos)
	angleRad = vtk.vtkMath.AngleBetweenVectors(line1DirectionVector, line2DirectionVector)
	angleDeg = vtk.vtkMath.DegreesFromRadians(angleRad)
	return angleDeg

line01Node='left_line'
line01Fids=['target','entry']
line02Node='acpc'
line02Fids=['pc','ac']
angleBetweenLineAndLine(line01Node, line01Fids, line02Node, line02Fids)

def angleBetweenFiducialPlaneAndSlice(fidName, fidList,sliceView):
	fidNode = list(slicer.util.getNodes(f'*{fidName}*').values())[0]
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


def angleBetweenLineAndPlane(P1,P2, planeNode, fidList):
	lineStartPos = P1
	lineEndPos = P2
	lineDirectionVector = (lineEndPos-lineStartPos)/np.linalg.norm(lineEndPos-lineStartPos)
	fidNode = slicer.util.getNode(planeNode)
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
	angleRad = vtk.vtkMath.AngleBetweenVectors(planeNormal, lineDirectionVector)
	angleDeg = vtk.vtkMath.DegreesFromRadians(angleRad)
	return angleDeg

planeNode='frame_center'
fidList=['midB','midE','midH']
P1=fc
P2=crossHairRAS
angleBetweenLineAndPlane(P1,P2, planeNode, fidList)


sliceView='vtkMRMLSliceNodeRed'
markupNode='LSTN_line'
startFid='target'
endFid='entry'
angleBetweenLineAndSlice(startFid, endFid, markupNode, sliceView)

sliceView='vtkMRMLSliceNodeGreen'
markupNode='left_line'
startFid='target'
endFid='entry'
angleBetweenLineAndSlice(startFid, endFid, markupNode, sliceView)

sliceView='vtkMRMLSliceNodeYellow'
markupNode='left_line'
startFid='target'
endFid='entry'
angleBetweenLineAndSlice(startFid, endFid, markupNode, sliceView)


sliceView='vtkMRMLSliceNodeRed'
markupNode='frame_center'
startFid='frame_center'
endFid='frame_center-2'
axial_angle=angleBetweenLineAndSlice(startFid, endFid, markupNode, sliceView)
axial_angle

sliceView='vtkMRMLSliceNodeGreen'
markupNode='frame_top_bottom'
startFid='right_C3_top'
endFid='right_C3_bot'
axial_angle=angleBetweenLineAndSlice(startFid, endFid, markupNode, sliceView)
90-axial_angle




fidName='desc-topbottom'
fidList=['ac', 'pc', 'mid1', 'mid2']
sliceView='vtkMRMLSliceNodeYellow'
angle2=180-angleBetweenFiducialPlaneAndSlice(fidName, fidList, sliceView)
print(f"IS angle is {angle2}")

fidName='desc-topbottom'
fidlistIgnore=['bar_A_top', 'bar_C_top','bar_I_top', 'bar_G_top']
sliceView='vtkMRMLSliceNodeRed'
angleX=angleBetweenFiducialPlaneAndSlice(fidName, fidlistIgnore,sliceView)
angleX

fidName='desc-topbottom'
fidlistIgnore=['bar_A_top', 'bar_A_bot','bar_C_top', 'bar_C_bot']
sliceView='vtkMRMLSliceNodeYellow'
angleY=angleBetweenFiducialPlaneAndSlice(fidName, fidlistIgnore,sliceView)
angleY

fidName='desc-topbottom'
fidlistIgnore=['bar_D_top', 'bar_D_bot','bar_F_top', 'bar_F_bot']
sliceView='vtkMRMLSliceNodeGreen'
angleZ=angleBetweenFiducialPlaneAndSlice(fidName, fidlistIgnore,sliceView)
angleZ

trans=vtk.vtkTransform()
if angleX>90:
	trans.RotateX(180-angleX)
else:
	trans.RotateX(-1*angleX)

if angleY>90:
	trans.RotateY(180-angleY)
else:
	trans.RotateY(-1*angleY)

if angleZ>90:
	trans.RotateZ(180-angleZ)
else:
	trans.RotateZ(-1*angleZ)

frame=slicer.vtkMRMLLinearTransformNode()
frame.SetName('frame')
slicer.mrmlScene.AddNode(frame)
frame.SetMatrixTransformToParent(trans.GetMatrix())




fidName='frame_top_bottom'
fidlistIgnore=['right_A1_bot','right_C1_bot','right_A3_bot','right_C3_bot']
sliceView='vtkMRMLSliceNodeRed'
axialAngel=angleBetweenFiducialPlaneAndSlice(fidName, fidlistIgnore,sliceView)
print(f"LR angle is {axialAngel}")


def getFrameCenter(bounds, nodeName):
	fidNode = slicer.util.getFirstNodeByClassByName('vtkMRMLMarkupsFiducialNode', nodeName)
	nOfFiduciallPoints = 0
	for i in range(fidNode.GetNumberOfFiducials()):
		if any(x in fidNode.GetNthFiducialLabel(i) for x in bounds):
			nOfFiduciallPoints +=1
	points = np.zeros([3, nOfFiduciallPoints])
	cnt=0
	for i in range(fidNode.GetNumberOfFiducials()):
		if any(x in fidNode.GetNthFiducialLabel(i) for x in bounds):
			fidNode.GetNthControlPointPositionWorld(i, points[:,cnt])
			cnt+=1
	planePosition = points.mean(axis=1)
	return planePosition

bounds = ['right_A3_top', 'right_A3_bot','right_C3_top', 'right_C3_bot',
			'right_A1_top', 'right_A1_bot','right_C1_top', 'right_C1_bot']
nodeName='frame_top_bottom'


frame_centerCoords=getFrameCenter(bounds, nodeName)


# This markups fiducial node specifies the center of rotation
centerOfRotationMarkupsNode=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')
n = centerOfRotationMarkupsNode.AddControlPoint(vtk.vtkVector3d(frame_centerCoords[0],frame_centerCoords[1],frame_centerCoords[2]))
centerOfRotationMarkupsNode.SetNthControlPointLabel(n, 'frame_center')
centerOfRotationMarkupsNode.SetNthControlPointLocked(n, True)


# This transform can be  edited in Transforms module
rotationTransformNode=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode')
rotationTransformNode.SetName('rotationTransformNode')


# This transform has to be applied to the image, model, etc.
finalTransformNode=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLTransformNode')
finalTransformNode.SetName('finalTransformNode')


def updateFinalTransform(unusedArg1=None, unusedArg2=None, unusedArg3=None):
    rotationMatrix = vtk.vtkMatrix4x4()
    rotationTransformNode.GetMatrixTransformToParent(rotationMatrix)
    rotationCenterPointCoord = [0.0, 0.0, 0.0]
    centerOfRotationMarkupsNode.GetNthControlPointPositionWorld(0, rotationCenterPointCoord)
    finalTransform = vtk.vtkTransform()
    finalTransform.Translate(rotationCenterPointCoord)
    finalTransform.Concatenate(rotationMatrix)
    finalTransform.Translate(-rotationCenterPointCoord[0], -rotationCenterPointCoord[1], -rotationCenterPointCoord[2])
    finalTransformNode.SetAndObserveMatrixTransformToParent(finalTransform.GetMatrix())


		
frameTopBot = slicer.util.getFirstNodeByClassByName('vtkMRMLMarkupsFiducialNode', 'sub-P062_space-leksellg_desc-topbottom_fids')
a_top = [0] * 3
c_bot = [0] * 3
d_top = [0] * 3
f_bot = [0] * 3
g_bot = [0] * 3
i_top = [0] * 3
for ifid in range(frameTopBot.GetNumberOfControlPoints()):
	if 'A_top' in frameTopBot.GetNthControlPointLabel(ifid):
		frameTopBot.GetNthControlPointPositionWorld(ifid, a_top)
	elif 'C_bot' in frameTopBot.GetNthControlPointLabel(ifid):
		frameTopBot.GetNthControlPointPositionWorld(ifid, c_bot)
	elif 'D_top' in frameTopBot.GetNthControlPointLabel(ifid):
		frameTopBot.GetNthControlPointPositionWorld(ifid, d_top)
	elif 'F_bot' in frameTopBot.GetNthControlPointLabel(ifid):
		frameTopBot.GetNthControlPointPositionWorld(ifid, f_bot)
	elif 'G_bot' in frameTopBot.GetNthControlPointLabel(ifid):
		frameTopBot.GetNthControlPointPositionWorld(ifid, g_bot)
	elif 'I_top' in frameTopBot.GetNthControlPointLabel(ifid):
		frameTopBot.GetNthControlPointPositionWorld(ifid, i_top)

midB = np.array(a_top) + norm_vec(a_top,c_bot) * ((mag_vec(a_top, c_bot) / 2)+1.3)
midE = (np.array(d_top) + norm_vec(d_top, f_bot) * (mag_vec(d_top, f_bot) / 2))+np.array([-1,1,-1.3])
midH = (np.array(i_top) + norm_vec(i_top, g_bot) * (mag_vec(i_top, g_bot) / 2))+np.array([-1,1,-1.3])

FC = np.array([(midB[0] + midH[0]) / 2, (midB[1] + midH[1]) / 2, (midB[2] + midH[2]) / 2])

frameCentNode=slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')
frameCentNode.SetAndObserveTransformNodeID(getNode('frame_center').GetParentTransformNode().GetID())
n = frameCentNode.AddControlPointWorld(vtk.vtkVector3d(FC[0],FC[1],FC[2]))
frameCentNode.SetNthControlPointLabel(n, 'frame_center')
frameCentNode.SetNthControlPointLocked(n, True)

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



tarCoords=[0]*3
getNode('left_line').GetNthControlPointPositionWorld(1, tarCoords)
FC=[0]*3
getNode('frame_center').GetNthControlPointPositionWorld(0, FC)

sliceFrameFids=getSliceFrameFids(np.round(tarCoords[2],1))
(mag_vec(sliceFrameFids[7,:],sliceFrameFids[8,:]) + mag_vec(sliceFrameFids[0,:],sliceFrameFids[1,:]))/2+40

CT_to_frame_coords(tarCoords, FC)



AC_ct=[0]*3
getNode('midline').GetNthControlPointPositionWorld(0, AC_ct)
PC_ct=[0]*3
getNode('midline').GetNthControlPointPositionWorld(1, PC_ct)
midpoint_ct=[0]*3
getNode('midline').GetNthControlPointPositionWorld(3, midpoint_ct)

n = frameCentNode.AddControlPointWorld(vtk.vtkVector3d(AC_ct[0],AC_ct[1],AC_ct[2]))
frameCentNode.SetNthControlPointLabel(n, 'ac')
frameCentNode.SetNthControlPointLocked(n, True)
n = frameCentNode.AddControlPointWorld(vtk.vtkVector3d(PC_ct[0],PC_ct[1],PC_ct[2]))
frameCentNode.SetNthControlPointLabel(n, 'pc')
frameCentNode.SetNthControlPointLocked(n, True)
n = frameCentNode.AddControlPointWorld(vtk.vtkVector3d(midpoint_ct[0],midpoint_ct[1],midpoint_ct[2]))
frameCentNode.SetNthControlPointLabel(n, 'mid')
frameCentNode.SetNthControlPointLocked(n, True)

AC_frame = CT_to_frame_coords(AC_ct, FC)
PC_frame = CT_to_frame_coords(PC_ct, FC)
midpoint_frame = CT_to_frame_coords(midpoint_ct, FC)

#midpoint_frame = np.array(midpoint_ct)
MCP_frame = np.array([(AC_frame[0]+PC_frame[0])/2, (AC_frame[1]+PC_frame[1])/2,(AC_frame[2]+PC_frame[2])/2])

print(f"Frame center is: {FC}")
print(f"Frame AC is: {AC_frame}")
print(f"Frame PC is: {PC_frame}")
print(f"Frame PC is: {midpoint_frame}")
print(f"Frame MCP is: {MCP_frame}")

IC = AC_frame-PC_frame
len_IC = np.sqrt((IC[0])**2+(IC[1])**2+(IC[2])**2)

e_y = np.array([IC[0]/len_IC, IC[1]/len_IC, IC[2]/len_IC])
w = np.array(midpoint_frame)-np.array(MCP_frame) 
crossp = np.cross(e_y, w)

if float(np.diff((MCP_frame[2], midpoint_frame[2]))) > 0:
	e_x = -1*(crossp/np.sqrt((crossp[0])**2+(crossp[1])**2+(crossp[2])**2))
else:
	e_x = (crossp/np.sqrt((crossp[0])**2+(crossp[1])**2+(crossp[2])**2))

e_z = np.cross(e_x, e_y)
#TP_frame = MP_frame + (target[0]*e_x) + (target[1]*e_y) + (target[2]*e_z)


rotx = np.arctan2(DirVec[1], DirVec[2])
roty = np.arctan2(DirVec[0]*np.cos(rotx), DirVec[2])
np.round(float((np.arccos(Xr[0]/dist))*180/np.pi),2)

import math
Xt=list(finalPTarget)
Xe=list(finalPEntry)

def Targets_to_Frame(Xt, Xe):
	Xt = np.array(Xt)
	Xe = np.array(Xe)
	Xr = np.array(Xe-Xt)
	dist = np.linalg.norm(Xr)
	phi=np.array([0.0,0.0])
	phi[0]=np.arccos(Xr[0]/dist)
	if Xr[1]!=0:
		phi[1]=np.arctan(Xr[2]/Xr[1])
	else:
		phi[1]=np.pi/2.0
	if phi[1]<0:
		phi[1]=np.pi+phi[1]
	return [math.degrees(phi[0]),math.degrees(phi[1])]

def applyTransformToPoints(transform, points, reverse=False):
	transformMatrix = vtk.vtkGeneralTransform()
	if reverse:
		slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(None, transform, transformMatrix)
	else:
		slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(transform, None, transformMatrix)
	finalPoints = transformMatrix.TransformPoint(points)
	return finalPoints

fc=[0]*3
getNode('frame_center').GetNthControlPointPositionWorld(0, fc)
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

entry=[0]*3
getNode('RSTN_line').GetNthControlPointPosition(1, entry)
target=[0]*3
getNode('RSTN_line').GetNthControlPointPosition(0, target)
entry=np.array(entry)
target=np.array(target)

arcAngle, ringAngle=Targets_to_Frame(target,entry)
arcAngle
ringAngle

sagAngle, axAngle=Targets_to_Frame(pc,ac)
sagAngle
axAngle
