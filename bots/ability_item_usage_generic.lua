
-- Called every frame. Responsible for issuing item usage actions.
function ItemUsageThinkOverride()

end

-- Called every frame. Responsible for issuing ability usage actions.
function AbilityUsageThinkOverride()

end

-- Called every frame. Responsible for issuing commands to the courier.
function CourierUsageThinkOverride()

end

-- Called every frame. Responsible for issuing a command to buyback.
function BuybackUsageThinkOverride()

end

-- Called every frame. Responsible for managing ability leveling.
function AbilityLevelUpThinkOverride()

end


ItemUsageThink = ItemUsageThinkOverride
AbilityUsageThink = AbilityUsageThinkOverride
CourierUsageThink = CourierUsageThinkOverride
BuybackUsageThink = BuybackUsageThinkOverride
AbilityLevelUpThink = AbilityLevelUpThinkOverride

print('Ability Item Usage Generic')