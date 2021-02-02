-- Called every frame. Responsible for selecting heroes for bots.
function ThinkOverride()
    if GetTeam() == TEAM_RADIANT then
        SelectHero(0, 'npc_dota_hero_antimage')
        SelectHero(1, 'npc_dota_hero_axe')
        SelectHero(2, 'npc_dota_hero_bane')
        SelectHero(3, 'npc_dota_hero_bloodseeker')
        SelectHero(4, 'npc_dota_hero_crystal_maiden')
    else
        SelectHero(5, 'npc_dota_hero_drow_ranger')
        SelectHero(6, 'npc_dota_hero_earthshaker')
        SelectHero(7, 'npc_dota_hero_juggernaut')
        SelectHero(8, 'npc_dota_hero_mirana')
        SelectHero(9, 'npc_dota_hero_nevermore')
    end
end

-- Called every frame prior to the game starting. Returns ten PlayerID-Lane pairs.
-- function UpdateLaneAssignmentsOverride()
--     return {
--         0 = LANE_TOP,
--         1 = LANE_TOP,
--         2 = LANE_MID,
--         3 = LANE_BOT,
--         4 = LANE_BOT,
--         --
--         5 = LANE_TOP,
--         6 = LANE_TOP,
--         7 = LANE_MID,
--         8 = LANE_BOT,
--         9 = LANE_BOT,
--     }
-- end

-- Called once, returns a table of player names.
function GetBotNamesOverride()
    return {
        'Blue',
        'Teal',
        'Purple',
        'Yellow',
        'Orange',
        'Pink',
        'Grey',
        'Aqua',
        'Green',
        'Brown'
    }
end

print('Hero Selection')
GetBotNames = GetBotNamesOverride
UpdateLaneAssignments = UpdateLaneAssignmentsOverride
Think = ThinkOverride
