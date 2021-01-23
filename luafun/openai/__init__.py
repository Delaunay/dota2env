import luafun.openai.five_model as five_model

import luafun.openai.mid_stich as mid_stich
import luafun.openai.mid_model as mid_model


game_config = {
    
}

states = {
    'openai-mid': mid_stich.FactionState,
    'openai-five': mid_stich.FactionState
}

stichers = {
    'openai-mid': mid_stich.apply_diff,
    'openai-five': mid_stich.apply_diff
}

models = {
    'openai-mid': mid_model.MidModel,
    'openai-five': five_model.FiveModel
}
