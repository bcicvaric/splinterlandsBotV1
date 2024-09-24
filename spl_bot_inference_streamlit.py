import streamlit as st
from spl_model import *

class GPTConfig:
    block_size: int = 64
    lineup_size: int = 7
    vocab_size: int = 648 # number of cards in the game
    n_layer: int = 3
    n_head: int = 1
    n_rules: int = 57
    n_embd: int = 64
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

#LOAD MODEL
gptconf = GPTConfig()
inference = GPT(gptconf)
inference = torch.load('spl_bot_model.pt')

def get_rule_id(rule, rules):
    try:
        return rules[rules['rule_name'] == rule]['rule_id'].values[0]
    except:
        return None

def get_state_np(mana, ruleset, inactive, rules, inactive_splinters, lineup=None, max_position=7):
    state = np.zeros(shape=(len(rules) - 1 + len(inactive_splinters) + 1 + max_position),dtype=int)

    # ADD MANA
    state[0] = int(mana)

    # ADD RULESET TO STATE
    if ruleset != 'Standard':

        ruleset_list = ruleset.split('|')
        for rule in ruleset_list:
            state[get_rule_id(rule, rules)] = 1

    # ADD INACTIVE TO STATE
    if inactive != "['']":

        inactive_list = inactive.strip("]['").split(',')
        for splinter in inactive_list:
            state[inactive_splinters[splinter] + len(rules) - 1] = 1

    # ADD LINEUP
    if lineup is not None:
        for i in range(7):
            state[len(state.columns) - 7 + i] = int(lineup[i])

    return state

def generate(model, x_0, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        output_shape = idx.shape[0]

        for i in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = model(x_0, idx)

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue

            idx[:, i] = idx_next.reshape(output_shape, )

        return idx

#INPUT CONSTANTS
rules = pd.read_csv('rules.csv')
inactive_splinters = {'Red':1,'Blue':2,'Green':3,'White':4,'Black':5,'Gold':6}
n_embd = 64
lineup_size = 7
batch_size = 1

def ids_to_names(lineup_ids):

    cards_df = get_all_cards()['name']
    #adding extra name for empty slot in lineups
    cards_df.loc[0] = 'none'

    f = lambda x: cards_df.loc[x]

    return f(lineup_ids)


def recommend_lineup():

    battle_rules = get_state_np(mana, ruleset, inactive, rules, inactive_splinters)

    x0 = torch.zeros(size=(1, 1, 64), dtype=torch.int)
    x0[0][0][0:len(battle_rules)] = torch.from_numpy(battle_rules)
    x1 = torch.zeros(size=(1, 7), dtype=torch.int)

    lineup_ids = generate(inference, x0, x1, max_new_tokens=7, top_k=3)[0].numpy()

    st.write(ids_to_names(lineup_ids))


# Streamlit app
st.title("Splinterlands Battle Recommender")
st.write('Instructions: input mana, ruleset & inactive splinters for a splinterlands battle')
mana = st.slider('Input mana',12,99)
inactive = st.text_input('Input inactive splinters',"['Red,Blue,Green,Black']")
ruleset = st.text_input('Input ruleset','Counterspell|Equalizer|Stampede')
st.button('Recommend team!',on_click=recommend_lineup)