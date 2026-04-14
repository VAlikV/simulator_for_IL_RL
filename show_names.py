import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("robot/scene.xml")
data = mujoco.MjData(model)

def list_objects(model, obj_type, count):
    for i in range(count):
        name = mujoco.mj_id2name(model, obj_type, i)
        print(f"{i:3d} | {name}")

# Использование:
# list_objects(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)
list_objects(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
# list_objects(model, mujoco.mjtObj.mjOBJ_SENSOR, model.nsensor)