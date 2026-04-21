Assignment 3, by William Zhao

Assignment Requirements
- Implemented Inverse Kinematics, using adolc to estimate the formula
- Implemented Forward Kinematics,
- Implemented Skinning, which deforms in real time to the bone control points moving.

Extra Credit
- Dynamic ADOLC iterations:
    - The number of iterations of the IK estimation depends on the average handle error.
    - If the expected handle position is far away from the actual handle position, the error rate is high and the number of iteratios increases.
- Tunable Damped IK (Tikhonov Regularization):
    - Implemented adjustable damping parameter for IK solver regularization.
    - Keyboard controls:
      - D: Increase damping (x1.5) - for smoother, more stable motion
      - F: Decrease damping (÷1.5) - for faster, more responsive motion
      - G: Toggle damping value display in console
    - Default damping: 1e-3 (good balance for most scenarios)

- FK Joint Manipulation (Forward Kinematics Direct Control):
    - Allow direct manipulation of joint rotations without IK constraints.
    - Keyboard controls:
      - K: Toggle FK manipulation mode on/off (default: IK mode)
      - =: Select next joint (cycles through all joints)
      - X: Rotate selected joint around X-axis
      - Y: Rotate selected joint around Y-axis
      - Z: Rotate selected joint around Z-axis
    - Mouse control: Drag vertically while in FK mode to rotate the selected joint around the chosen axis
- Recording
    - P to start recording
    - P to stop recording
