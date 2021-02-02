Dota 2 ML Bots
==============

Why dota2 is the ultimate RL environment

* Long Time Horizons
* Partially Observed State
* High dimensional action and observation spaces

Like in real life you need to
1. Choose the relevant data to solve the problem and discard unnecessary data
2. Choose the correct rewards balance early rewards and late rewards
3. Encode the action

# Assets

Most assets located in `luafun/game/resources` are owned by Valve.

# Platform

* works best on linux
* there is a bug in windows where state are being dropped randomly (unknown reason)

#  Install

1. Download [Anaconda 3][1]
2. Start > Anaconda Prompt (anaconda)
3. git clone https://github.com/Delaunay/LuaFun.git
4. cd LuaFun
4. Install Dependencies
    * conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
    * pip install -r requirements.txt
    * pip install -e .
5. change the bot folder
    * DOTA_PATH=/media/setepenre/local/SteamLibraryLinux/steamapps/common/dota\ 2\ beta
    * cd $DOTA_PATH/game/dota/scripts/vscripts
    * ln -s ~/work/LuaFun/bots/ bots
6. Run the simulation
7. Use you browser to see how it is going
    http://localhost:5000/draw/radiant

[1]: https://www.anaconda.com/products/individual



# The Game

From OpenAI

* 30 FPS         (~0.033 s per frame)
* 45 Minutes
* Every 4 Frame  (~0.133 s per decision, 450 actions per minute)
* Reaction Time is 0.266 seconds
* 20300 steps 

1. Observe 4 Frames
2. Use 4 Frames to decide action
3. Take action in the nect 4 Frames


1. The action space (Appendix F)
    Allows the player to change the game state in order to increase the probability of winning overtime

2. The observed game state (Appendix E)
    Allows the player to observe the current game in order to make a decision

3. Objective Function (Appendix G)
    Rewards the player for an action increasing its probability of winning
 
 
## The Action Space

Filter the action space to present a set of actions to choose from

  * Abilities (4+)
  * 6 Items
  * Aghs
  * Shard
  * 1 Neutral Item
  * 1 TP
  * Attack
  * Move
  * Cancel
  * Purchase
  * buyback
  * take outpost
  * courier shield
  * courier speed burst
  
3 Action Parameters

* 4 Dim Frame Delay (useless)
* Unit Selection 189 dim (189 = 5 + 5 + 15 + 2 * 82)
* Offset (position in a 9x9 square)

* Action Type
    * No Target
    * Point Target
    * Unit Target
    * Unit Offset Target
    * Teleport Target
    * Ward Target
    
## The Game State

> 1,200 categorical values and 14,534 continuous/boolean values.


* Global
    * Time Since Game start
    * is_day
    * Time until next day or Night
    * Time until next Soldier Spawn
    * Time until next bounty Spawn
    * Time until next runes Spawn
    * Time until next Neutral Spawn
    * Time since seen Courrier
    * Roshan Current Max HP
    * is_roshan_alive
    * is_roshan_dead
    * does_roshan_drop_cheese
    * does_roshan_drop_refresher
    * Glyph Cooldown Radiant
    * Glyph Cooldown Dire
    * Item Stock Count
        * Gem
        * Smoke
        * Wards
        * Infused RainDrops
        * Agh Shard

* Units: 189 (5 heroes, 30 creeps, 21 buildings, 30 wards, 5 couriers)
    * position.x
    * position.y
    * position.z
    * angle.cos
    * angle.sin
    * is_attacking
    * time_since_last_attack
    * max_health
    * heath_[t-16:now]
    * attack damage
    * attack speed
    * physical resistance
    * invulnerable due to glyph
    * glyph timer
    * movement speed
    * is_allied
    * is_neutral
    * animation cycle time
    * ETA of incoming projectile
    * vector to me dx                       [e]
    * vector to me dy                       [e]
    * vector to me dz                       [e]
    * am_I_attacking_it                     [e]
    * is_it_attacking_me                    [e]
    * ETA of projectile from unit to me     [e]
    * unit type
    * current animation

* Heroes (10)
    * is_alive
    * death_count
    * hero_in_sight
    * hero_last_seen
    * hero_teleporting
    * teleport_target.x
    * teleport_target.y
    * teleport_channel_time
    * respawn_time
    * current_gold
    * level
    * mana_max
    * mana_current
    * mana_regen
    * health_regen
    * magic_resistance
    * strength
    * agi
    * intel
    * invisible
    * is_using_ability
    * is_allied
    * is_enemy
    * is_creep
    * is_hero
    * buyback_available
    * buyback_cooldown
    * buyback_cost
    * empty_backpack_slots
    * empty_inventory_slots
    * lane assignment.top
    * lane assignment.mid
    * lane assignment.bot
    * nearby_terrain (14x14 grid)

* Nearby Map (8x8) [e]
    * elevation
    * passable
    * allied creep density
    * enemy creep density
    * area of effect
    * spells in effect
    
* Previously Sampled Action 310 [e]
    * Offset (3x2x9)
    * Unit Target Embedding     (128)
    * Primary Action Embedding  (128)

* Hero Modifiers (10 x 10)
    * remaining_duration
    * stack_count
    * modifier_name

* Item Modifiers (10 x 16)
    * location.inventory
    * location.backpack
    * location.stash
    * charges
    * is_on_cooldown
    * cooldown_time
    * is_disabled_by_swap
    * item_swap_cooldown
    * toggled_state
    * item_state.str
    * item_state.agi
    * item_state.int
    * item_state.none
    * item_name

* Per Abilities (10 x 6)
    * cooldown_time 
    * in_use
    * castable
    * level1
    * level2
    * level3
    * level4

* Per Pickup (6)
    is_there
    is_not_there
    state_unknown
    location.x
    location.y
    distance_from_hero_0
    distance_from_hero_1
    distance_from_hero_2
    distance_from_hero_3
    distance_from_hero_4
    distance_from_hero_5
    distance_from_hero_6
    distance_from_hero_7
    distance_from_hero_8
    distance_from_hero_9

* Minimap (10x10)
    is_visible
    allied creeps
    enemy creeps
    enemy heroes
    allied wards
    enemy wards
    cell.x
    cell.y
    cell.id ?


* Normalize the data by the running mean and std clip((obs - mean) / std, -5, 5)

## The Objective Function

    
    Hero Reward    := (Team Reward + Solo Reward) * 0.6 ^ (T / 10 mins)
    
    Reward         := (1 - p) * Solo Reward + p * (Sum of Allied Reward) / 4
    
    Dire Reward    := Sum(Reward) - Radiant Reward
    Radiant Reward := Sum(Reward) - Dire Reward


Normalize the reward overtime as the ability to farm increase


### Rewards

Name                | Reward    | Type
--------------------|-----------|-------
Win                 |    5      | Team
Hero Death          |   -1      | Solo
Courier Death       |   -2      | Team 
XP Gained           |    0.002  | Solo 
Gold Gained         |    0.006  | Solo
Gold Spent          |    0.0006 | Solo
Health Changed      |    2      | Solo | %of health = ( x + 1 - (1 - x) ^ 4) / 2
Mana Changed        |    0.75   | Solo
Killed Hero         |   -0.6    | Solo
Last Hit            |   -0.16   | Solo | Reduce the reward since we already got a bit chunk from exp & gold
Deny                |    0.15   | Solo
Gained Aegis        |    5      | Team
Ancient HP Change   |    5      | Team
Megas Unlocked      |    4      | Team
T1 Tower            |    2.25   | Team
T2 Tower            |    3      | Team
T3 Tower            |    4.5    | Team
T4 Tower            |    2.25   | Team | 2/3 = building health + 1/3 on destroy
Outpost             |    2.25   | Team
Barracks            |    6      | Team | 2/3 = building health + 1/3 on destroy
Lane Assign         |   -0.15   | Solo


# Embeddings

## Unit Target Embedding

Unit Info -> f -> Vector(128) -> F > Unit Info


## Action Embedding

* Primary Action Embedding  (128)

## Ability & Item Embedding

We see Item as Ability you can buy/Equip

# The ML Model

The ML model takes a state as input and outputs an action.
The goal is to train the model to make decision that increase its likelyness of winning

Maximize the probability of winning through a reward function.
Reward function is a zero sum game (reward earned by the opposite team diminish the overall reward)

## OpenAI Original Model

* Shared LSTM block of 4096 units
* Connected to separate fully Connected layers
* Final model had 158,502,815 parameters

## Inference Engine

# Optimization processs

* PPO with GAE
* 120 samples, eachwith 16 timestep
* Adam Optimizer with truncated back propagation over 16 timesteps (64 Frames or 2.13 seconds)


# Integration with DOt2 Engine (Appendix K)


Message sent/received to/from a gRPC server that fowards it to LUA which makes the decision


# Transfer learning

int luaopen_genericbot (lua_State* L)







* 5 Ally Heroes
* 5 Enemy Heroes
* 25 buildinds
* 30 creeps
* 30 wards
* 5 couriers
* 15 Neutrals

### Hero State

  * Health
  * Mana
  * Gold
  * Experience
  * Position
  * Abilities (4+)
  * 6 Item Slots
  * Aghs
  * Shard
  * 1 Neutral Item
  * 1 TP
  * Stats
        * Attack    (Speed, Damage, Range, Spell Amp, Mana Regen)
        * Armor     (Amror, Phys, Magic, Status, Evasion, Health Regen)
        * Move Speed
        * Strength
        * Agility
        * Intelligence

### Non Hero State

  * Health
  * Mana
  

