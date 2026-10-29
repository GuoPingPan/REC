"""Original native environment primitives plus actual cross-scene transition."""
from pathlib import Path
import json
import numpy as np
from AirVLN.src.vlnce_src.env import AirVLNENV
from AirVLN.src.common.param import args
from AirVLN_utils.env_utils import getPoseAfterMakeAction


def values(p):
    return np.array([p.position.x_val, p.position.y_val, p.position.z_val,
                     p.orientation.x_val, p.orientation.y_val, p.orientation.z_val, p.orientation.w_val])


if __name__ == '__main__':
    env = AirVLNENV(batch_size=1, split=args.split, dataset='aerialvln-s')
    episodes = len(env.data)
    assert 1 <= episodes <= 2
    results = []
    try:
        for index in range(episodes):
            env.next_minibatch()
            obs = env.reset()
            rgb = obs[0][0]['rgb']
            assert rgb.shape == (224, 224, 3) and float(rgb.std()) > 1
            from PIL import Image
            Image.fromarray(rgb[..., ::-1]).save('native_smoke_rgb.png')
            checks = []
            for action in range(8):
                env._setEpisodes()
                env.update_measurements()
                expected = getPoseAfterMakeAction(env.sim_states[0].pose, action)
                env.makeActions([action])
                actual = env.simulator_tool.airsim_clients[0][0].simGetVehiclePose(vehicle_name='Drone_1')
                error = float(np.max(np.abs(values(expected) - values(actual))))
                assert error < 1e-3, (action, error)
                outputs = env.get_obs()
                checks.append({'action': action, 'pose_max_abs_error': error, 'official_metrics': outputs[0][3]})
            results.append({'episode_id': str(env.batch[0]['episode_id']), 'scene_id': env.batch[0]['scene_id'],
                            'rgb_shape': list(rgb.shape), 'checks': checks})
            print('SCENE_CHECKED', env.batch[0]['scene_id'], flush=True)
    finally:
        env.simulator_tool.closeScenes()
        env.delete_VectorEnvUtil()
    report = {'passed': True, 'episodes': results, 'cross_scene_transition_checked': episodes == 2,
              'checks': [x for result in results for x in result['checks']], 'policy_inference_run': False,
              'scope': 'Original environment and native primitive deployment; no learned policy evaluated.'}
    if episodes == 1:
        report.update({k: results[0][k] for k in ['episode_id', 'scene_id', 'rgb_shape']})
    Path('native_smoke_result.json').write_text(json.dumps(report, indent=2))
    print('NATIVE_SMOKE_PASSED', len(report['checks']), flush=True)
