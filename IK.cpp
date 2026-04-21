#include "IK.h"
#include "FK.h"
#include "minivectorTemplate.h"
#include <Eigen/Dense>
#include <adolc/adolc.h>
#include <iostream>
using namespace std;

// CSCI 520 Computer Animation and Simulation
// Jernej Barbic and Yijing Li

namespace
{

constexpr double PI = 3.141592653589793238462643383279502884;

// Converts degrees to radians.
template<typename real>
inline real deg2rad(const real & deg) { return deg * real(PI / 180.0); }

template<typename real>
void euler2Rotation(const real angle[3], real R[9], RotateOrder order)
{
  Mat3<real> RX = Mat3<real>::getElementRotationMatrix(0, deg2rad(angle[0]));
  Mat3<real> RY = Mat3<real>::getElementRotationMatrix(1, deg2rad(angle[1]));
  Mat3<real> RZ = Mat3<real>::getElementRotationMatrix(2, deg2rad(angle[2]));

  Mat3<real> RMat;
  switch(order)
  {
    case RotateOrder::XYZ:
      RMat = RZ * RY * RX;
      break;
    case RotateOrder::YZX:
      RMat = RX * RZ * RY;
      break;
    case RotateOrder::ZXY:
      RMat = RY * RX * RZ;
      break;
    case RotateOrder::XZY:
      RMat = RY * RZ * RX;
      break;
    case RotateOrder::YXZ:
      RMat = RZ * RX * RY;
      break;
    case RotateOrder::ZYX:
      RMat = RX * RY * RZ;
      break;
  }
  RMat.convertToArray(R);
}

// Performs forward kinematics, using the provided "fk" class.
// This is the function whose Jacobian matrix will be computed using adolc.
// numIKJoints and IKJointIDs specify which joints serve as handles for IK:
//   IKJointIDs is an array of integers of length "numIKJoints"
// Input: numIKJoints, IKJointIDs, fk, eulerAngles (of all joints)
// Output: handlePositions (world-coordinate positions of all the IK joints; length is 3 * numIKJoints)
template<typename real>
void forwardKinematicsFunction(
    int numIKJoints, const int * IKJointIDs, const FK & fk,
    const std::vector<real> & eulerAngles, std::vector<real> & handlePositions)
{
  const int numJoints = fk.getNumJoints();

  std::vector<Mat3<real>> localRotations(numJoints), globalRotations(numJoints);
  std::vector<Vec3<real>> localTranslations(numJoints), globalTranslations(numJoints);

  for (int jointID = 0; jointID < numJoints; jointID++)
  {
    const Vec3d & restTranslation = fk.getJointRestTranslation(jointID);
    const Vec3d & jointOrient = fk.getJointOrient(jointID);
    RotateOrder rotateOrder = fk.getJointRotateOrder(jointID);

    real jointOrientAngles[3] = { real(jointOrient[0]), real(jointOrient[1]), real(jointOrient[2]) };
    real jointEuler[3] = {
      eulerAngles[3 * jointID + 0],
      eulerAngles[3 * jointID + 1],
      eulerAngles[3 * jointID + 2]
    };

    Mat3<real> jointOrientRotation;
    euler2Rotation(jointOrientAngles, jointOrientRotation.data(), RotateOrder::XYZ);
    Mat3<real> animatedRotation;
    euler2Rotation(jointEuler, animatedRotation.data(), rotateOrder);

    localRotations[jointID] = jointOrientRotation * animatedRotation;
    localTranslations[jointID] = Vec3<real>(restTranslation.data());
  }

  for (int updateIndex = 0; updateIndex < numJoints; updateIndex++)
  {
    int jointID = fk.getJointUpdateOrder(updateIndex);
    int parentID = fk.getJointParent(jointID);

    if (parentID < 0)
    {
      globalRotations[jointID] = localRotations[jointID];
      globalTranslations[jointID] = localTranslations[jointID];
    }
    else
    {
      globalRotations[jointID] = globalRotations[parentID] * localRotations[jointID];
      globalTranslations[jointID] = globalRotations[parentID] * localTranslations[jointID] + globalTranslations[parentID];
    }
  }

  handlePositions.resize(3 * numIKJoints);
  for (int i = 0; i < numIKJoints; i++)
  {
    int jointID = IKJointIDs[i];
    handlePositions[3 * i + 0] = globalTranslations[jointID][0];
    handlePositions[3 * i + 1] = globalTranslations[jointID][1];
    handlePositions[3 * i + 2] = globalTranslations[jointID][2];
  }
}

} // end anonymous namespaces

IK::IK(int numIKJoints, const int * IKJointIDs, FK * inputFK, int adolc_tagID)
{
  this->numIKJoints = numIKJoints;
  this->IKJointIDs = IKJointIDs;
  this->fk = inputFK;
  this->adolc_tagID = adolc_tagID;

  FKInputDim = fk->getNumJoints() * 3;
  FKOutputDim = numIKJoints * 3;

  train_adolc();
}

void IK::train_adolc()
{
  const auto tag = static_cast<short>(adolc_tagID);
  trace_on(tag);

  vector<adouble> x(FKInputDim);
  for (int i = 0; i < FKInputDim; i++)
    x[i] <<= 0.0;

  vector<adouble> y;
  forwardKinematicsFunction<adouble>(numIKJoints, IKJointIDs, *fk, x, y);

  vector<double> output(FKOutputDim, 0.0);
  for (int i = 0; i < FKOutputDim; i++)
    y[i] >>= output[i];

  trace_off();
}

void IK::doIK(const Vec3d * targetHandlePositions, Vec3d * jointEulerAngles)
{
  const int numJoints = fk->getNumJoints();
  const int inputDim = FKInputDim;
  const int outputDim = FKOutputDim;
  const auto tag = static_cast<short>(adolc_tagID);

  vector<double> x(inputDim);
  for (int jointID = 0; jointID < numJoints; jointID++)
  {
    x[3 * jointID + 0] = jointEulerAngles[jointID][0];
    x[3 * jointID + 1] = jointEulerAngles[jointID][1];
    x[3 * jointID + 2] = jointEulerAngles[jointID][2];
  }

  vector<double> y(outputDim, 0.0);
  vector<double> residual(outputDim, 0.0);
  vector<double> jacobianStorage(outputDim * inputDim, 0.0);
  vector<double *> jacobianRows(outputDim, nullptr);
  for (int i = 0; i < outputDim; i++)
    jacobianRows[i] = &jacobianStorage[i * inputDim];

  Eigen::MatrixXd J(outputDim, inputDim);
  Eigen::VectorXd r(outputDim);
  const double damping = dampingFactor;

  ::function(tag, outputDim, inputDim, x.data(), y.data());
  double avgHandleError = 0.0;
  for (int handleID = 0; handleID < numIKJoints; handleID++)
  {
    double dx = targetHandlePositions[handleID][0] - y[3 * handleID + 0];
    double dy = targetHandlePositions[handleID][1] - y[3 * handleID + 1];
    double dz = targetHandlePositions[handleID][2] - y[3 * handleID + 2];
    avgHandleError += sqrt(dx * dx + dy * dy + dz * dz);
  }
  if (numIKJoints > 0)
    avgHandleError /= numIKJoints;

  int maxIterations = 10;
  if (avgHandleError > 0.20)
    maxIterations = 60;
  else if (avgHandleError > 0.05)
    maxIterations = 35;
  else if (avgHandleError > 0.01)
    maxIterations = 20;

  for (int iter = 0; iter < maxIterations; iter++)
  {
    ::function(tag, outputDim, inputDim, x.data(), y.data());
    ::jacobian(tag, outputDim, inputDim, x.data(), jacobianRows.data());

    for (int i = 0; i < outputDim; i++)
    {
      int component = i % 3;
      residual[i] = targetHandlePositions[i / 3][component] - y[i];
      r[i] = residual[i];
      for (int j = 0; j < inputDim; j++)
        J(i, j) = jacobianStorage[i * inputDim + j];
    }

    Eigen::MatrixXd A = J.transpose() * J;
    A.diagonal().array() += damping * damping;
    Eigen::VectorXd b = J.transpose() * r;
    Eigen::VectorXd dx = A.ldlt().solve(b);

    if (!dx.allFinite())
      break;

    double stepNorm = dx.norm();
    if (stepNorm < 1e-8)
      break;

    for (int j = 0; j < inputDim; j++)
      x[j] += dx[j];

    if (b.norm() < 1e-6)
      break;
  }

  for (int jointID = 0; jointID < numJoints; jointID++)
  {
    jointEulerAngles[jointID][0] = x[3 * jointID + 0];
    jointEulerAngles[jointID][1] = x[3 * jointID + 1];
    jointEulerAngles[jointID][2] = x[3 * jointID + 2];
  }
}
