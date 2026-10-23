import click
import gymnasium as gym

import logging
import numpy as np
import os
import stable_baselines3
import stable_baselines3.common
import stable_baselines3.common.base_class
import torch

from pathlib import Path
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import (
    StopTrainingOnNoModelImprovement,
    StopTrainingOnRewardThreshold,
    EvalCallback,
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

# while not called directly, we need to import this so the environments are registered
import envs

# STR-91: per-ticket artifact layout, shared with the evaluation scripts so the training and
# evaluation sides agree on where a ticket's artifacts go
from visibility.experiment_tracking import Experiment, JiraTicket, TICKET_OPTION_HELP
from visibility.reward_visibility import RewardTermsCallback

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def resolve_model_file(ctx: dict, environment: str, suffix: str = ".zip") -> str:
    """
    The model to load: --model when given, otherwise this ticket's best model.

    STR-91: the default used to be the shared models/<EnvId>_<ALGO>/ folder, which held
    whichever run finished last.
    """
    model_file = ctx.obj['MODEL_PATH']
    if model_file is None:
        experiment = ctx.obj['EXPERIMENT']
        model_dir = experiment.model_dir(environment, ctx.obj['ALGORITHM_NAME'])
        model_file = str(model_dir / f"best_model{suffix}")

    if not os.path.isfile(model_file):
        raise RuntimeError(f"Could not open model file: {model_file}")

    return model_file


def algorithm_factory(algorithm_name: str, env: gym.Env, log_dir: str) -> BaseAlgorithm:
    """
    Factory method to create algorithm. Will also include any extra params needed for this
    particular case (balancing robot)
    """
    if algorithm_name == "DDPG":
        policy_kwargs = dict(
            net_arch=dict(pi=[300, 200], qf=[200, 150])
        )
        action_noise = NormalActionNoise(
            mean=np.zeros(2),
            sigma=0.1 * np.ones(2)
        )
        model = stable_baselines3.DDPG(
            "MlpPolicy",
            env=env,
            verbose=1,
            device=DEVICE,
            tensorboard_log=log_dir,
            policy_kwargs=policy_kwargs,
            action_noise=action_noise
        )
        return model
    elif algorithm_name == "PPO":
        model = stable_baselines3.PPO(
            "MlpPolicy",
            env=env,
            verbose=1,
            device=DEVICE,
            tensorboard_log=log_dir
        )
        return model
    else:
        algorithm_class = getattr(stable_baselines3, algorithm_name, None)
        if algorithm_class is None:
            return None
        model = algorithm_class(
            'MlpPolicy',
            env,
            verbose=1,
            device=DEVICE,
            tensorboard_log=log_dir
        )
        return model


@click.command(name="convert", help="Convert a PyTorch model to ONNX format")
@click.option('-e', '--environment', required=True, type=str, help="id of Gymnasium environment (eg; ReferenceBaselineTrainingEnv)")
@click.pass_context
def convert(ctx: dict, environment: str):
    """ Converts model to ONNX format """
    env = gym.make(environment, render_mode='rgb_array')

    algorithm_name = ctx.obj['ALGORITHM_NAME']

    model_file = resolve_model_file(ctx, environment)

    logger.info(f"Converting model to ONNX")
    logger.info(f"Algorithm: {algorithm_name}")
    logger.info(f"Environment: {environment}")
    logger.info(f"Model: {model_file}")

    algorithm_class = getattr(stable_baselines3, algorithm_name, None)
    model = algorithm_class.load(model_file, env=env)

    # Get the underlying PyTorch model (policy network)
    pytorch_model = model.policy

    # The model must be in evaluation mode for export. If it's in training mode, behaviors
    # like dropout and batch normalization will yield non-deterministic results.
    pytorch_model.eval()

    # Create a dummy input based on the observation space
    dummy_input = torch.zeros(1, *env.observation_space.shape)

    input_path = Path(model_file)
    output_path = input_path.with_suffix('.onnx')

    logger.info(f"Output file: {str(output_path)}")
    # Export the model to ONNX format
    torch.onnx.export(
        pytorch_model,
        dummy_input,
        str(output_path),
        opset_version=11,
        input_names=['input'],
        output_names=['output']
    )


@click.command(name="test", help="Test the current model")
@click.option('-e', '--environment', required=True, type=str, help="id of Gymnasium environment (eg; ReferenceBaselineTrainingEnv)")
@click.option('--show-io', is_flag=True, default=False, help="log model inputs and outputs")
@click.option('--show-i', is_flag=True, default=False, help="log model inputs to std out in Python array syntax")
@click.pass_context
def test(ctx: dict, environment: str, show_io: bool, show_i: bool):
    """ Test a model by running in MuJoCo interactively """
    env = gym.make(environment, render_mode='human')

    algorithm_name = ctx.obj['ALGORITHM_NAME']

    model_file = resolve_model_file(ctx, environment)

    logger.info(f"Starting test simulation")
    logger.info(f"Algorithm: {algorithm_name}")
    logger.info(f"Environment: {environment}")
    logger.info(f"Model: {model_file}")

    algorithm_class = getattr(stable_baselines3, algorithm_name, None)
    model = algorithm_class.load(model_file, env=env)

    run_loop_count = 0
    terminated_loop_count = 0
    obs = env.reset()[0]
    while True:
        action, _ = model.predict(obs)
        print(f"Action: {action}")
        if show_io and run_loop_count % 30 == 0:
            logger.info(str(list(obs) + list(action)))
        if show_i and run_loop_count % 30 == 0:
            logger.info(str(list(obs)) + ",")

        obs, _, terminated, truncated, _ = env.step(action)

        if (terminated or truncated) and terminated_loop_count > 200:
            terminated_loop_count = 0
            obs = env.reset()[0]
            # break
        if terminated or truncated:
            terminated_loop_count +=1

        run_loop_count += 1


@click.command(name="test-onnx", help="Test an ONNX model")
@click.option('-e', '--environment', required=True, type=str, help="id of Gymnasium environment (eg; ReferenceBaselineTrainingEnv)")
@click.option('--show-io', is_flag=True, default=False, help="log model inputs and outputs")
@click.pass_context
def test_onnx(ctx: dict, environment: str, show_io: bool):
    """ Test an ONNX model by running in MuJoCo interactively """
    import onnx
    import onnxruntime as ort

    env = gym.make(environment, render_mode='human')

    algorithm_name = ctx.obj['ALGORITHM_NAME']

    model_file = resolve_model_file(ctx, environment, suffix=".onnx")

    logger.info(f"Starting ONNX test simulation")
    logger.info(f"Algorithm: {algorithm_name}")
    logger.info(f"Environment: {environment}")
    logger.info(f"Model: {model_file}")

    onnx_model = onnx.load(model_file)
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(model_file)
    # Get input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    run_loop_count = 0
    obs = env.reset()[0]
    while True:
        input_tensor = np.array([obs], dtype=np.float32)
        onnx_outputs = session.run([output_name], {input_name: input_tensor})[0][0]

        if show_io and run_loop_count % 30 == 0:
            logger.info(str(list(obs) + list(onnx_outputs)))

        obs, _, terminated, truncated, _ = env.step(onnx_outputs)

        if terminated or truncated:
            break

        run_loop_count += 1


@click.command(name="test-tflite", help="Test a tflite model")
@click.option('-e', '--environment', required=True, type=str, help="id of Gymnasium environment (eg; ReferenceBaselineTrainingEnv)")
@click.option('--show-io', is_flag=True, default=False, help="log model inputs and outputs")
@click.option('--show-i', is_flag=True, default=False, help="log model inputs to std out in Python array syntax")
@click.pass_context
def test_tflite(ctx: dict, environment: str, show_io: bool, show_i: bool):
    """ Test a tflite model by running in MuJoCo interactively """
    import tensorflow as tf

    env = gym.make(environment, render_mode='human')

    algorithm_name = ctx.obj['ALGORITHM_NAME']

    model_file = resolve_model_file(ctx, environment, suffix=".tflite")

    logger.info(f"Starting tflite test simulation")
    logger.info(f"Algorithm: {algorithm_name}")
    logger.info(f"Environment: {environment}")
    logger.info(f"Model: {model_file}")

    interpreter = tf.lite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    run_loop_count = 0
    obs = env.reset()[0]
    while True:
        input_tensor = np.array([obs], dtype=np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        if show_io and run_loop_count % 30 == 0:
            logger.info(str(list(obs) + list(output_data)))
        if show_i and run_loop_count % 30 == 0:
            logger.info(str(list(obs)) + ",")

        obs, _, terminated, truncated, _ = env.step(output_data)

        if terminated or truncated:
            break

        run_loop_count += 1


@click.command(name="test-tflite-quant", help="Test a quantized tflite model")
@click.option('-e', '--environment', required=True, type=str, help="id of Gymnasium environment (eg; ReferenceBaselineTrainingEnv)")
@click.pass_context
def test_tflite_quant(ctx: dict, environment: str):
    """ Test a tflite model by running in MuJoCo interactively """
    import tensorflow as tf

    env = gym.make(environment, render_mode='human')

    algorithm_name = ctx.obj['ALGORITHM_NAME']

    model_file = ctx.obj['MODEL_PATH']
    if model_file is None:
        raise RuntimeError(f"Must provide model file")

    if not os.path.isfile(model_file):
        raise RuntimeError(f"Could not open model file: {model_file}")

    logger.info(f"Starting tflite quantized test simulation")
    logger.info(f"Algorithm: {algorithm_name}")
    logger.info(f"Environment: {environment}")
    logger.info(f"Model: {model_file}")

    interpreter = tf.lite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    # there's only one input tensor
    input_detail = input_details[0]
    # get the quantization input params
    input_scale, input_zero_point = input_detail['quantization']
    logger.info("Quantization parameters")
    logger.info(f"Input scale: {input_scale}")
    logger.info(f"Input zero point {input_zero_point}")

    output_details = interpreter.get_output_details()
    # unique to the PPO policy, which has multiple outputs
    # the second output is the one that includes the actions needed
    output_detail = output_details[1]
    output_scale, output_zero_point = output_detail['quantization']
    logger.info(f"Output scale: {output_scale}")
    logger.info(f"Output zero point {output_zero_point}")

    run_loop_count = 0
    obs = env.reset()[0]
    while True:
        # convert observation values to quantized values using parameters from the tflite
        # quantized model
        obs_quant = [
            (np.round(obs_value / input_scale) + input_zero_point)
            for obs_value in obs
        ]

        # need to make sure the quantized values are within the range of an int8, otherwise
        # the values get wrapped and -129 becomes +127! Which is obviously bad for the
        # robots balance
        obs_quant = np.clip(obs_quant, a_min = -128, a_max = 127)

        input_tensor = np.array([obs_quant], dtype=np.int8)
        logger.info(input_tensor)

        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output_data_quant = interpreter.get_tensor(output_details[1]['index'])[0]

        logger.info("             " + str(output_data_quant))

        # convert quantized model outputs back to floats as expected by the sim
        output_data = np.array(
            [
                output_scale * (output_data_quant[0].astype(np.float32) - output_zero_point),
                output_scale * (output_data_quant[1].astype(np.float32) - output_zero_point),
            ],
            dtype=np.float32
        )

        obs, _, terminated, truncated, _ = env.step(output_data)

        if terminated or truncated:
            break

        run_loop_count += 1


def write_to_arduino(data:str, ser: "serial.Serial"):
    """
    Write a string to the Arduino via serial.
    """
    if ser.isOpen():
        ser.write(data.encode())  # Convert string to bytes and send
        ser.write(b'\n')          # Send newline to signal end of input
        # print(f"Sent to Arduino: {data}")
    else:
        print("Error: Serial port is not open.")

def read_from_arduino(ser: "serial.Serial") -> str:
    """
    Read the response from the Arduino.
    """
    if ser.isOpen():
        # Read available data from Arduino
        response = ser.readline().decode('utf-8').strip()  # Read response and decode to string
        print(f"Arduino says: {response}")
        return response
    else:
        print("Error: Serial port is not open.")
        return None


@click.command(name="test-tflite-arduino", help="Test a quantized tflite model on the arduino")
@click.option('-e', '--environment', required=True, type=str, help="id of Gymnasium environment (eg; ReferenceBaselineTrainingEnv)")
@click.pass_context
def test_tflite_arduino(ctx: dict, environment: str):
    """ Test a tflite model by running in MuJoCo interactively """
    import tensorflow as tf

    env = gym.make(environment, render_mode='human')

    algorithm_name = ctx.obj['ALGORITHM_NAME']

    model_file = ctx.obj['MODEL_PATH']
    if model_file is None:
        raise RuntimeError(f"Must provide model file")

    if not os.path.isfile(model_file):
        raise RuntimeError(f"Could not open model file: {model_file}")

    logger.info(f"Starting tflite quantized test simulation")
    logger.info(f"Algorithm: {algorithm_name}")
    logger.info(f"Environment: {environment}")
    logger.info(f"Model: {model_file}")

    arduino_port = '/dev/cu.usbmodem158747801'  # Replace with your port
    baud_rate = 115200
    timeout = 1  # Set a timeout for reading from the Arduino

    import serial

    # Initialize serial connection
    ser = serial.Serial(arduino_port, baud_rate, timeout=timeout)

    # We never run inference on this model here, as that's done on
    # the arduino in this command. However, we do need to load it up
    # to extract the quantization params (assumes loaded model is the
    # same as the one running on the arduino)
    interpreter = tf.lite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    # there's only one input tensor
    input_detail = input_details[0]
    # get the quantization input params
    input_scale, input_zero_point = input_detail['quantization']
    logger.info("Quantization parameters")
    logger.info(f"Input scale: {input_scale}")
    logger.info(f"Input zero point {input_zero_point}")

    output_details = interpreter.get_output_details()
    # unique to the PPO policy, which has multiple outputs
    # the second output is the one that includes the actions needed
    output_detail = output_details[1]
    output_scale, output_zero_point = output_detail['quantization']
    logger.info(f"Output scale: {output_scale}")
    logger.info(f"Output zero point {output_zero_point}")

    run_loop_count = 0
    obs = env.reset()[0]
    while True:
        # convert observation values to quantized values using parameters from the tflite
        # quantized model
        obs_quant = [
            (np.round(obs_value / input_scale) + input_zero_point)
            for obs_value in obs
        ]

        # need to make sure the quantized values are within the range of an int8, otherwise
        # the values get wrapped and -129 becomes +127! Which is obviously bad for the
        # robots balance
        obs_quant = np.clip(obs_quant, a_min = -128, a_max = 127)

        input_tensor = np.array([obs_quant], dtype=np.int8)
        # logger.info(input_tensor)

        # write_to_arduino(data=",".join(map(str, input_tensor[0].tolist())), ser=ser)
        logger.info(obs)
        write_to_arduino(data=",".join(map(str, obs)), ser=ser)

        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()

        arduino_resp = read_from_arduino(ser)
        # output_data_quant = [int(s) for s in arduino_resp.split(',')]
        output_data_quant = [float(s) for s in arduino_resp.split(',')]

        logger.info(output_data_quant)

        # convert quantized model outputs back to floats as expected by the sim
        # output_data = np.array(
        #     [
        #         output_scale * (output_data_quant[0].astype(np.float32) - output_zero_point),
        #         output_scale * (output_data_quant[1].astype(np.float32) - output_zero_point),
        #     ],
        #     dtype=np.float32
        # )

        obs, _, terminated, truncated, _ = env.step(output_data_quant)

        if terminated or truncated:
            break

        run_loop_count += 1


@click.command(name="train", help="Train a model with a given environment")
@click.option('-e', '--environment', required=True, type=str, help="id of Gymnasium environment (eg; ReferenceBaselineTrainingEnv)")
@click.option('-t', '--timesteps', default=int(1e10), type=int, help="max training timesteps (default: run until stopped)")
@click.pass_context
def train(ctx: dict, environment: str, timesteps: int):
    """ Train a model by with a given MuJoCo environment """

    algorithm_name = ctx.obj['ALGORITHM_NAME']
    # STR-91: all artifacts below go to experiments/<TICKET>/, not shared logs/models/movies
    experiment = ctx.obj['EXPERIMENT']
    experiment.make_training_dirs()

    env = gym.make(environment, render_mode='rgb_array')
    env = Monitor(env)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(experiment.movies),
        video_length=0,
        episode_trigger = lambda x: x % 50 == 0,
    )

    # STR-105: normalize observation variables
    env = gym.wrappers.NormalizeObservation(env=env, epsilon=1e-8)

    logger.info(f"Starting training process")
    logger.info(f"Algorithm: {algorithm_name}")
    logger.info(f"Environment: {environment}")
    logger.info(f"Experiment: {experiment.ticket}")
    logger.info(f"Artifacts: {experiment.root}")

    model_file = ctx.obj['MODEL_PATH']
    if model_file is None:
        # no model given, so create a new one
        model = algorithm_factory(
            algorithm_name=algorithm_name,
            env=env,
            log_dir=str(experiment.logs)
        )
        logger.info(f"Model: starting with new model")
    elif os.path.isfile(model_file):
        # start with an existing model
        algorithm_class = getattr(stable_baselines3, algorithm_name, None)
        if algorithm_class is None:
            raise RuntimeError(f"Couldn't find algorithm {algorithm_name}")
        model = algorithm_class.load(model_file, env=env, tensorboard_log=str(experiment.logs))
        logger.info(f"Model: starting with {model_file}")
    else:
        raise RuntimeError(f"Model file {model_file} does not exist")

    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=6000, verbose=1)
    stop_train_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5,
        min_evals=10000,
        verbose=1
    )

    # STR-91: best model and checkpoints land in this ticket's folder, so a later experiment
    # cannot overwrite them - a trained model was lost that way before this layout existed
    model_dir = experiment.model_dir(environment, algorithm_name)

    eval_callback = EvalCallback(
        env,
        eval_freq=20000,
        callback_on_new_best=callback_on_best,
        callback_after_eval=stop_train_callback,
        verbose=1,
        best_model_save_path=str(model_dir),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=40000,
        verbose=2,
        save_path=str(model_dir),
        name_prefix=f"{environment}_{algorithm_name}_cp_"
    )

    callbacks = [checkpoint_callback, eval_callback]
    # only envs with configurable reward weights (currently ReferenceBaselineTrainingEnv) expose reward terms
    reward_weights = getattr(env.unwrapped, "reward_weights", None)
    if reward_weights is not None:
        callbacks.append(RewardTermsCallback(reward_weights))

    # STR-91: record what produced this run so the folder is self-describing
    experiment.write_metadata(
        command="train",
        environment=environment,
        algorithm=algorithm_name,
        timesteps=timesteps,
        device=DEVICE,
        starting_model=model_file,
        reward_weights=reward_weights,
    )

    try:
        model.learn(
            total_timesteps=timesteps,
            tb_log_name=f"{environment}_{algorithm_name}",
            callback=CallbackList(callbacks)
        )
    finally:
        # Closing the env makes RecordVideo flush the clip it is part way through, so the last
        # movie is actually watchable. In a finally block because these runs normally end with
        # Ctrl-C rather than by hitting total_timesteps. Movies are local only - post anything
        # worth keeping to the Jira ticket.
        try:
            env.close()
        except Exception:
            logger.warning("Could not close the environment cleanly", exc_info=True)


@click.group()
@click.option(
    '-a',
    '--algorithm',
    required=True,
    type=str,
    help="Stable Baseline3 algorithm (eg; A2C, DDPG, DQN, PPO, SAC, TD3)"
)
@click.option(
    '-m',
    '--model',
    default=None,
    type=click.Path(exists=False),
    help="Path to model file"
)
# STR-91: the ticket is required - it is the experiment's identity in the log
@click.option(
    '-out',
    '--output_path',
    'ticket',
    required=True,
    type=JiraTicket(),
    help=TICKET_OPTION_HELP
)
@click.pass_context
def cli(ctx: dict, algorithm: str, model: str | os.PathLike, ticket: str):
    algorithm_class = getattr(stable_baselines3, algorithm, None)
    if algorithm_class is None:
        raise RuntimeError(f"Could not find Stable Baselines3 algorithm for: {algorithm}")

    # ensure that ctx.obj exists and is a dict (in case `cli()` is called
    # by means other than the `if` block below)
    ctx.ensure_object(dict)
    ctx.obj['ALGORITHM_NAME'] = algorithm
    ctx.obj['MODEL_PATH'] = model
    ctx.obj['EXPERIMENT'] = Experiment(ticket)  # STR-91


cli.add_command(convert)
cli.add_command(test)
cli.add_command(test_onnx)
cli.add_command(test_tflite)
cli.add_command(test_tflite_quant)
cli.add_command(test_tflite_arduino)
cli.add_command(train)


if __name__ == '__main__':
    # STR-91: artifact folders are created per experiment, once the ticket is known (see Experiment)
    cli(obj={})

