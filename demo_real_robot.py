"""
Usage:
(robodiff)$ python demo_real_robot.py -o <demo_save_dir> --robot_ip <ip_of_ur5>

Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Use joystick buttons:
- Button 0 (A) to start recording
- Button 1 (B) to stop recording  
- Button 4 (LEFT_BUMPER) to exit program
"""

# %%
import time
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import os
import scipy.spatial.transform as st
from diffusion_policy.real_world.real_env import RealEnv
from diffusion_policy.common.precise_sleep import precise_wait
from controllers.joystick.joystick_control import JoystickInterface, JoystickAxis, JoystickButton

@click.command()
@click.option('--output', '-o', required=True, help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', required=True, help="UR5's IP address e.g. 192.168.0.204")
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
def main(output, robot_ip, vis_camera_idx, init_joints, frequency, command_latency):
    output_prefix = "/tmp/diffusion_policy"
    output = os.path.join(output_prefix, output)
    dt = 1/frequency
    with SharedMemoryManager() as shm_manager:
        joystick = JoystickInterface()
        with RealEnv(
                output_dir=output, 
                robot_ip=robot_ip, 
                # recording resolution
                obs_image_resolution=(640,480),
                frequency=frequency,
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                record_raw_video=True,
                # number of threads per camera view for video recording (H.264)
                thread_per_video=3,
                # video recording quality, lower is better (but slower).
                video_crf=21,
                shm_manager=shm_manager
            ) as env:
            if not joystick.connect_device(0):
                print("No joystick found. Exiting.")
                return
            print("Joystick connected.")
            
            cv2.setNumThreads(1)

            # # realsense exposure
            # env.realsense.set_exposure(exposure=120, gain=0)
            # # realsense white balance
            # env.realsense.set_white_balance(white_balance=5900)

            time.sleep(1.0)
            print('Ready!')
            state = env.get_robot_state()
            target_pose = state['TargetTCPPose']
            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False
            delete_episode = 0
            while not stop:
                # calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                # pump obs
                obs = env.get_obs()

                # handle joystick button presses
                joystick_state = joystick.read_state(0)
                
                # Check for button presses
                if joystick_state.buttons.get(JoystickButton.A.value, False):
                    # Start recording (Button 0)
                    env.start_episode(t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                    is_recording = True
                    delete_episode = 0
                    print('Recording!')
                elif joystick_state.buttons.get(JoystickButton.B.value, False):
                    # Stop recording (Button 1)
                    env.end_episode()
                    is_recording = False
                    delete_episode = 0
                    print('Stopped.')
                elif joystick_state.buttons.get(JoystickButton.LEFT_BUMPER.value, False):
                    # Exit program (Button 4)
                    stop = True
                elif joystick_state.buttons.get(JoystickButton.Y.value, False):
                    delete_episode += 1
                    print(f"[zyu]you sure delete_episode? {delete_episode}")
                    if delete_episode == 2:
                        env.drop_episode()
                        delete_episode = 0
                elif joystick_state.buttons.get(JoystickButton.MANU.value, False):
                    env.max_pos_speed = 0.01
                else:
                    env.max_pos_speed = 0.1
                precise_wait(t_sample)  
                
                # visualize
                vis_img = obs[f'camera_{vis_camera_idx}'][-1,:,:,::-1].copy()
                episode_id = env.replay_buffer.n_episodes
                text = f'Episode: {episode_id}'
                if is_recording:
                    text += ', Recording!'
                    print(f"[zyu] recording")
                else:
                    print(f"[zyu] stopping")
                # Take out for script server useage
                # cv2.putText(
                #     vis_img,
                #     text,
                #     (10,30),
                #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #     fontScale=1,
                #     thickness=2,
                #     color=(255,255,255)
                # )
                # cv2.imshow('default', vis_img)
                precise_wait(t_sample) 
            

                # get teleop command
                x_pos = -joystick_state.axes.get(JoystickAxis.LEFT_X.value, 0.0) * (env.max_pos_speed / frequency)
                y_pos = joystick_state.axes.get(JoystickAxis.LEFT_Y.value, 0.0) * (env.max_pos_speed / frequency)
                if joystick_state.buttons.get(JoystickButton.BACK.value, False):
                    z_pos = -(env.max_pos_speed / frequency)
                elif joystick_state.buttons.get(JoystickButton.START.value, False):
                    z_pos = (env.max_pos_speed / frequency)
                else:
                    z_pos = 0.0
                dpos = np.array([x_pos, y_pos, z_pos])
                # drot_xyz = sm_state[3:] * (env.max_rot_speed / frequency)
                
                # if not sm.is_button_pressed(0):
                #     # translation mode
                #     drot_xyz[:] = 0
                # else:
                #     dpos[:] = 0
                # if not sm.is_button_pressed(1):
                #     # 2D translation mode
                #     dpos[2] = 0    
                drot_xyz = np.array([0.0, 0.0, 0.0])
                drot = st.Rotation.from_euler('xyz', drot_xyz)
                target_pose[:3] += dpos
                target_pose[3:] = (drot * st.Rotation.from_rotvec(
                    target_pose[3:])).as_rotvec()
                target_pose[3:] = np.array([0.0, -np.pi, 0.0])
                target_pose[:2] = np.clip(target_pose[:2], [-0.30, 0.40], [0.38, 0.90])
                # execute teleop command
                print(f"[zyu] target_pose: {target_pose}")
                env.exec_actions(
                    actions=[target_pose], 
                    timestamps=[t_command_target-time.monotonic()+time.time()],
                    stages=None)
                precise_wait(t_cycle_end)
                iter_idx += 1

# %%
if __name__ == '__main__':
    main()
