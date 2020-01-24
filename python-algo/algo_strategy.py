import gamelib
import random
import math
import warnings
from sys import maxsize
import json
import itertools
import random
import heapq
from myutil import div, create_line

"""
Most of the algo code you write will be in this file unless you create new
modules yourself. Start by modifying the 'on_turn' function.

Advanced strategy tips: 

  - You can analyze action frames by modifying on_action_frame function

  - The GameState.map object can be manually manipulated to create hypothetical 
  board states. Though, we recommended making a copy of the map to preserve 
  the actual current map state.
"""

class AlgoStrategy(gamelib.AlgoCore):
    def __init__(self):
        super().__init__()
        seed = random.randrange(maxsize)
        random.seed(seed)
        gamelib.debug_write('Random seed: {}'.format(seed))

    def on_game_start(self, config):
        """ 
        Read in config and perform any initial setup here 
        """
        gamelib.debug_write('Configuring your custom algo strategy...')
        self.config = config
        global FILTER, ENCRYPTOR, DESTRUCTOR, PING, EMP, SCRAMBLER, BITS, CORES
        FILTER = config["unitInformation"][0]["shorthand"]
        ENCRYPTOR = config["unitInformation"][1]["shorthand"]
        DESTRUCTOR = config["unitInformation"][2]["shorthand"]
        PING = config["unitInformation"][3]["shorthand"]
        EMP = config["unitInformation"][4]["shorthand"]
        SCRAMBLER = config["unitInformation"][5]["shorthand"]
        BITS = 1
        CORES = 0
        # This is a good place to do initial setup
        self.scored_on_locations = []

        self.passes = []

        self.plan = { "aggression": 0.5, "attack_waves": [] }

        self.history_cap = 5
        self.history_gamestate = []
        self.history_map_stationary = []
        self.damage_map = [[0 for y in range(28)] for x in range(28)]
        self.damage_decay = 0.9
        self.damage_priority = []
        self.attack_follow = False
        
    def on_turn(self, turn_state):
        """
        This function is called every turn with the game state wrapper as
        an argument. The wrapper stores the state of the arena and has methods
        for querying its state, allocating your current resources as planned
        unit deployments, and transmitting your intended deployments to the
        game engine.
        """
        game_state = gamelib.GameState(self.config, turn_state)
        gamelib.debug_write('Performing turn {} of your custom algo strategy'.format(game_state.turn_number))
        game_state.suppress_warnings(True)  #Comment or remove this line to enable warnings.

        self.history_gamestate.append(game_state)

        self.passes = [self.evaluate_board, self.get_stats, self.create_strategy]
        
        while len(self.passes)>0:
            f = self.passes[0]
            f(self.history_gamestate[-1], self.passes, self.plan)
            self.passes = self.passes[1:]

        if len(self.history_gamestate) > self.history_cap:
            self.history_gamestate = self.history_gamestate[1:]
            self.history_map_stationary = self.history_map_stationary[1:]
            assert len(self.history_gamestate) == len(self.history_map_stationary)
                
        # self.plan = { "aggression": self.plan.aggression, "attack_waves": self.plan.attack_waves }
        
        self.history_gamestate[-1].submit_turn()

    """
    NOTE: All the methods after this point are part of the sample starter-algo
    strategy and can safely be replaced for your custom algo.
    """
    
    def get_stats(self, game_state, passes, plan):

        plan["mybits"] = game_state.get_resources(BITS)
        plan["mycores"] = game_state.get_resources(CORES)
        plan["myhealth"] = game_state.my_health
        
        plan["ratio_bits"] = div(game_state.get_resource(BITS,1), game_state.get_resource(BITS))
        plan["ratio_cores"] = div(game_state.get_resource(CORES,1), game_state.get_resource(CORES))
        plan["ratio_health"] = div(game_state.enemy_health, game_state.my_health)

    def evaluate_board(self, game_state, passes, plan):

        map_stationary = [[None for y in range(28)] for x in range(28)]

        assert len(map_stationary) == 28
        assert len(map_stationary[0]) == 28
        
        for location in game_state.game_map:
            if game_state.contains_stationary_unit(location):
                for unit in game_state.game_map[location]:
                    if unit is not None and unit.player_index == 0 and gamelib.is_stationary(unit.unit_type):
                        map_stationary[location[0]][location[1]] = unit

        self.history_map_stationary.append(map_stationary)
        
        #get damage for own units
        if len(self.history_map_stationary)<2:
            return

        for x in range(game_state.ARENA_SIZE):
            for y in range(game_state.ARENA_SIZE):
                
                self.damage_map[x][y] = self.damage_decay * float(self.damage_map[x][y])

                # gamelib.debug_write("len(self.history_map_stationary): {}".format(len(self.history_map_stationary)))
                
                assert len(self.history_map_stationary) >= 2

                if self.history_map_stationary[-2][x][y] is not None:
                    
                    if map_stationary[x][y] is not None:
                        delta_health = self.history_map_stationary[-2][x][y].health - map_stationary[x][y].health
                    else:
                        delta_health = self.history_map_stationary[-2][x][y].health
                        
                    self.damage_map[x][y] += (1.0-self.damage_decay) * float(delta_health)
                            
        q = []
        for x in range(game_state.ARENA_SIZE):
            for y in range(game_state.ARENA_SIZE):
                heapq.heappush(q, (-self.damage_map[x][y], (x,y)))
                
        self.damage_priority = q
        
        # gamelib.debug_write("damage_priority: {}".format(heapq.nsmallest(5,q)))

    def create_strategy(self, game_state, passes, plan):
        
        def build_defense_basic(game_state, passes, plan):

            # filter_locations = create_line(start=[1,13],end=[11,3])
            # filter_locations.extend(create_line(start=[26,13],end=[16,3]))
            # filter_locations.extend(create_line(start=[13,7],end=[13,7]))
            # filter_locations.extend(create_line(start=[14,7],end=[14,8]))

            filter_locations = create_line(start=[1,13],end=[7,9])
            filter_locations.extend(create_line(start=[7,9],end=[21,13]))
            filter_locations.extend(create_line(start=[26,13],end=[16,3]))
            
            _, locations_spawn_filter = game_state.attempt_spawn(FILTER, filter_locations)

            destructor_locations = []
            # destructor_locations = create_line(start=[0,13],end=[0,13])
            destructor_locations = create_line(start=[0,13],end=[0,13])
            destructor_locations.extend(create_line(start=[27,13],end=[27,13]))
            destructor_locations.extend(create_line(start=[21,12],end=[17,10]))

            destructor_locations.extend(create_line(start=[10,8],end=[16,10])[::3])
                        
            # destructor_locations.extend(create_line(start=[13,5],end=[13,6]))
            # destructor_locations.extend(create_line(start=[14,5],end=[14,6]))
            # destructor_locations.extend(create_line(start=[13,5],end=[13,5]))
            # destructor_locations.extend(create_line(start=[14,5],end=[14,5]))

            _, locations_spawn_destructor = game_state.attempt_spawn(DESTRUCTOR, destructor_locations)

            cost_upgrade_destructor = game_state.type_cost(DESTRUCTOR, upgrade=True)
            cost_upgrade_filter = game_state.type_cost(FILTER, upgrade=True)

            target_cost_upgrade = 0.5 * float(game_state.get_resource(CORES))

            l = int(len(self.damage_priority)/2)
            upgrade_priority = heapq.nsmallest(l,self.damage_priority)

            for (k,v) in upgrade_priority:
                if target_cost_upgrade < 0.:
                    break
                if k < 0: # more negative means more damage soaked
                    units = game_state.game_map[[v[0],v[1]]]
                    for u in units:
                        if u is not None and u.player_index == 0 and gamelib.is_stationary(u.unit_type) and not u.upgraded:
                            c = game_state.type_cost(u.unit_type, upgrade=True)[CORES]
                            target_cost_upgrade = target_cost_upgrade - float(c)
                            game_state.attempt_upgrade([v[0],v[1]])
        
        def build_attack(game_state, passes, plan):

            # locs_info_spawn = create_line(start=[12,1],end=[13,0])
            # locs_info_spawn.extend(create_line(start=[15,1],end=[14,0]))

            locs_info_spawn = create_line(start=[6,7],end=[7,6])

            if game_state.get_resource(BITS) > 1:
                num_ping = int(max(0.1*float(game_state.number_affordable(SCRAMBLER)),1.))
                sel = random.choice(locs_info_spawn)
                game_state.attempt_spawn(SCRAMBLER, sel, num_ping)

            # if self.attack_follow is True:
            #     num_ping = int(max(0.1*float(game_state.number_affordable(PING)),1.))
            #     sel = random.choice(locs_info_spawn)
            #     game_state.attempt_spawn(PING, sel, num_ping)

            # self.attack_follow = False           
                
            # if game_state.get_resource(BITS) > 11:
            #     num_emp = game_state.number_affordable(EMP)
            #     sel = random.choice(locs_info_spawn)
            #     game_state.attempt_spawn(EMP, sel, num_emp)
            #     self.attack_follow = True

            if game_state.get_resource(BITS) > 15:
                num_emp = game_state.number_affordable(PING)
                sel = random.choice(locs_info_spawn)
                game_state.attempt_spawn(PING, sel, num_emp)
                self.attack_follow = True

        def build_defense_random(game_state, passes, plan):

            # encrytor_locations = create_line(start=[12,1],end=[12,1])
            encrytor_locations = create_line(start=[7,8],end=[13,2])
            encrytor_locations.extend(create_line(start=[15,1],end=[15,1]))
            game_state.attempt_spawn(ENCRYPTOR, encrytor_locations)
            
            destructor_locations = []
            # for i in range(9,19,4):
            #     destructor_locations.extend(create_line(start=[i,7],end=[i,7]))

            cost_destructor = game_state.type_cost(DESTRUCTOR)[CORES]

            num = int((0.75 * float(game_state.get_resource(CORES))) / float(cost_destructor))
            
            # l = int(len(self.damage_priority)/2)
            upgrade_priority = heapq.nsmallest(int(num/2),self.damage_priority)

            upgrade_priority = map(lambda a: [a,a,a], upgrade_priority)
            
            import itertools
            upgrade_priority = list(itertools.chain.from_iterable(upgrade_priority))
            
            # choices = random.choices(upgrade_priority, k=num)
            # gamelib.debug_write("choices: {}".format(choices))
            i = 0
            retry = 0
            while i < num and len(upgrade_priority)>i:
                k,v = upgrade_priority[i]
                # if k < 0: # more negative means more damage soaked
                x = random.randrange(v[0]-3,v[0]+3+1)
                y = random.randrange(v[1]-3,v[1]+3+1)
                if (x,y) not in plan["attack_path_reserve"]:
                    new_destructor = (create_line(start=[x,y],end=[x,y]))
                    num_spawned, _ = game_state.attempt_spawn(DESTRUCTOR, new_destructor)
                    if num_spawned > 0:
                        gamelib.debug_write("new destruct spawn: {}".format((x,y)))
                        i += 1
                        retry = 0
                        #spawn a filter
                        x = random.randrange(x-1,x+1+1)
                        y = random.randrange(y-1,y+1+1)
                        if (x,y) not in plan["attack_path_reserve"]:
                            _, _ = game_state.attempt_spawn(FILTER, (create_line(start=[x,y],end=[x,y])))
                    else:
                        retry += 1
                else:
                    if retry > 10:
                        i += 1
                        retry = 0
                    else:
                        retry += 1
                # else:
                #     break
            
            # i = 0
            # while i < num:
                
            #     side = random.randrange(0,2)
            #     if side == 0:
            #         x = random.randrange(0,4+1)
            #         y = random.randrange(10,13+1)
            #     else:
            #         x = random.randrange(22,27+1)
            #         y = random.randrange(10,13+1)
            #     if (x,y) not in plan["attack_path_reserve"]:
            #         destructor_locations.extend(create_line(start=[x,y],end=[x,y]))
            #         i += 1

            if(len(destructor_locations)>0):
                game_state.attempt_spawn(DESTRUCTOR, destructor_locations)

            cost_upgrade_destructor = game_state.type_cost(DESTRUCTOR, upgrade=True)
            cost_upgrade_filter = game_state.type_cost(FILTER, upgrade=True)

            target_cost_upgrade = 0.5 * float(game_state.get_resource(CORES))

            l = int(len(self.damage_priority)/2)
            upgrade_priority = heapq.nsmallest(l,self.damage_priority)

            for (k,v) in upgrade_priority:
                if target_cost_upgrade < 0.:
                    break
                if k < 0: # more negative means more damage soaked
                    units = game_state.game_map[[v[0],v[1]]]
                    for u in units:
                        if u is not None and u.player_index == 0 and gamelib.is_stationary(u.unit_type) and not u.upgraded:
                            c = game_state.type_cost(u.unit_type, upgrade=True)[CORES]
                            target_cost_upgrade = target_cost_upgrade - float(c)
                            game_state.attempt_upgrade([v[0],v[1]])


        def reserve_attack_path(game_state, passes, plan):
            plan["attack_path_reserve"] = {}
            # locs = create_line(start=[13,0],end=[13,4])
            # locs.extend(create_line(start=[14,0],end=[14,4]))
            # locs.extend(create_line(start=[12,4],end=[12,13]))
            # locs.extend(create_line(start=[15,4],end=[15,13]))

            locs = create_line(start=[25,13],end=[15,3])
            locs.extend(create_line(start=[24,13],end=[15,4]))
            locs.extend(create_line(start=[6,7],end=[13,0]))
            locs.extend(create_line(start=[7,7],end=[14,0]))
            locs.extend(create_line(start=[14,0],end=[15,4]))

            for i in locs:
                plan["attack_path_reserve"][(i[0],i[1])] = i
        
        passes.append(reserve_attack_path)
        passes.append(build_defense_basic)
        passes.append(build_attack)
        passes.append(build_defense_random)

    def on_action_frame(self, turn_string):
        """
        This is the action frame of the game. This function could be called 
        hundreds of times per turn and could slow the algo down so avoid putting slow code here.
        Processing the action frames is complicated so we only suggest it if you have time and experience.
        Full doc on format of a game frame at: https://docs.c1games.com/json-docs.html
        """
        # Let's record at what position we get scored on
        state = json.loads(turn_string)
        events = state["events"]
        breaches = events["breach"]
        for breach in breaches:
            location = breach[0]
            unit_owner_self = True if breach[4] == 1 else False
            # When parsing the frame data directly, 
            # 1 is integer for yourself, 2 is opponent (StarterKit code uses 0, 1 as player_index instead)
            if not unit_owner_self:
                gamelib.debug_write("Got scored on at: {}".format(location))
                self.scored_on_locations.append(location)
                gamelib.debug_write("All locations: {}".format(self.scored_on_locations))


if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()
