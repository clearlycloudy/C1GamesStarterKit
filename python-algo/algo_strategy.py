import gamelib
import random
import math
import warnings
from sys import maxsize
import json
import itertools
import random
import heapq
from myutil import div, create_line, create_gaussian_distr, generate_sample_from_gaussian

import numpy as np
import torch
from scipy import fftpack
from scipy import signal
from scipy import misc
# from sklearn import mixture

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
        # a = np.ones((10,1))
        # b = np.ones((10,1))
        # c = np.sum(a * b, axis=0)
        # gamelib.debug_write('numpy test: {}'.format(c))

        a = torch.randn(2, 2)
        a = ((a * 3) / (a - 1))
        gamelib.debug_write(a.requires_grad)
        a.requires_grad_(True)
        gamelib.debug_write(a.requires_grad)
        b = (a * a).sum()
        gamelib.debug_write(b.grad_fn)
        b.backward()
        gamelib.debug_write(a.grad)

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

        self.plan = { "aggression": 0.5, "attack_waves": [], "currently_reserving_attacking": False, "attack_path_reserve": {}, "can_attack": False }

        self.history_cap = 5
        self.history_gamestate = []
        self.history_map_stationary = []
        # self.damage_map = [[0 for y in range(28)] for x in range(28)]
        self.damage_map = np.zeros((28,28))
        self.damage_decay = 0.5
        self.damage_priority = []

        self.enemy_map = np.zeros((28,28))
        
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
        
        self.enemy_map = np.zeros((28,28))
        
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
                    if unit is not None and unit.player_index == 1 and gamelib.is_stationary(unit.unit_type):
                        self.enemy_map[location[0]][location[1]] += float(unit.health)

        gamelib.debug_write("enemy_map: {}".format(self.enemy_map))

        t = np.linspace(-5, 5, 30)
        # bump = np.exp(-0.4*t**2)
        bump = np.exp(-0.6*t**2)
        bump /= np.trapz(bump) # normalize the integral to 1

        # make a 2-D kernel out of it
        kernel = bump[:, np.newaxis] * bump[np.newaxis, :]

        filter_enemy_map = signal.convolve2d(self.enemy_map, kernel, boundary='symm', mode='same')

        if "attack_path_reserve_x_coord" not in plan or self.plan["attack_path_reserve"] is False:

            if filter_enemy_map[3,15] < filter_enemy_map[24,15]:
                plan["attack_path_reserve_x_coord"] = 3
            else:
                plan["attack_path_reserve_x_coord"] = 24
        
            
        self.history_map_stationary.append(map_stationary)
        
        #get damage for own units
        if len(self.history_map_stationary)<2:
            return

        temp = np.zeros((28,28))
        
        for x in range(game_state.ARENA_SIZE):
            for y in range(game_state.ARENA_SIZE):

                # gamelib.debug_write("len(self.history_map_stationary): {}".format(len(self.history_map_stationary)))
                
                assert len(self.history_map_stationary) >= 2

                if self.history_map_stationary[-2][x][y] is not None:
                    
                    if map_stationary[x][y] is not None:
                        delta_health = self.history_map_stationary[-2][x][y].health - map_stationary[x][y].health
                    else:
                        delta_health = self.history_map_stationary[-2][x][y].health

                    temp[x,y] = delta_health

        damage_bias = np.zeros((28,28))
        # damage_bias[26,12] = 10
        # damage_bias[1,12] = 10
        self.damage_map = (self.damage_map * self.damage_decay + (1.0-self.damage_decay) * temp) + damage_bias

        t = np.linspace(-5, 5, 30)
        bump = np.exp(-0.4*t**2)
        bump /= np.trapz(bump) # normalize the integral to 1

        # make a 2-D kernel out of it
        kernel = bump[:, np.newaxis] * bump[np.newaxis, :]

        # #smoothing filter
        filtered = signal.convolve2d(self.damage_map, kernel, boundary='symm', mode='same')

        filter_enemy_map = signal.convolve2d(self.enemy_map, kernel, boundary='symm', mode='same')
        
        # kernel_ft = fftpack.fft2(kernel, shape=self.damage_map.shape[:2], axes=(0, 1))

        # # convolve
        # img_ft = fftpack.fft2(self.damage_map, axes=(0, 1))
        # # the 'newaxis' is to match to color direction
        # img2_ft = kernel_ft[:, :, np.newaxis] * img_ft
        # img2 = fftpack.ifft2(img2_ft, axes=(0, 1)).real

        # gamelib.debug_write(self.damage_map)
        # gamelib.debug_write("filtered")
        # gamelib.debug_write(filtered)
                                            
        q = []
        for x in range(game_state.ARENA_SIZE):
            for y in range(game_state.ARENA_SIZE):
                # heapq.heappush(q, (-self.damage_map[x][y], (x,y)))
                heapq.heappush(q, (-filtered[x][y], (x,y)))
                
        self.damage_priority = q
        
        # gamelib.debug_write("damage_priority: {}".format(heapq.nsmallest(5,q)))

    def create_strategy(self, game_state, passes, plan):
        
        def build_defense_basic(game_state, passes, plan):

            gamelib.debug_write("attack_path_reserve: {}".format(plan["attack_path_reserve"]))
            # filter_locations = create_line(start=[1,12],end=[20,12])
            # # filter_locations.extend(create_line(start=[26,12],end=[22,8]))
            # filter_locations.extend(create_line(start=[26,13],end=[21,8]))
            # filter_locations.extend(create_line(start=[22,8],end=[16,8]))
            
            filter_locations = []
            # filter_locations.extend(create_line(start=[2,13],end=[25,13]))
            filter_locations.extend(create_line(start=[4,12],end=[23,12]))

            for i in filter_locations:
                x = i[0]
                y = i[1]
                if (x,y) not in plan["attack_path_reserve"]:
                    placement = [[x,y]]
                    num_spawned, _ = game_state.attempt_spawn(FILTER, placement)
                
            # _, locations_spawn_filter = game_state.attempt_spawn(FILTER, filter_locations)

            destructor_locations = []
            
            destructor_locations = create_line(start=[0,13],end=[0,13])
            destructor_locations.extend(create_line(start=[27,13],end=[27,13]))

            destructor_locations.extend(create_line(start=[3,12],end=[3,12]))
            destructor_locations.extend(create_line(start=[24,12],end=[24,12]))
            
            destructor_locations.extend(create_line(start=[1,13],end=[1,13]))
            destructor_locations.extend(create_line(start=[26,13],end=[26,13]))
            
            destructor_locations.extend(create_line(start=[4,12],end=[4,12]))

            destructor_locations.extend(create_line(start=[23,12],end=[23,12]))
            
            # destructor_locations.extend(create_line(start=[20,11],end=[20,11]))
            
            # destructor_locations.extend(create_line(start=[24,10],end=[24,10]))
            # destructor_locations.extend(create_line(start=[1,13],end=[1,13]))
            # destructor_locations.extend(create_line(start=[26,13],end=[26,13]))

            for i in destructor_locations:
                x = i[0]
                y = i[1]
                if (x,y) not in plan["attack_path_reserve"]:
                    placement = [[x,y]]
                    num_spawned, _ = game_state.attempt_spawn(DESTRUCTOR, placement)
                    
            # _, locations_spawn_destructor = game_state.attempt_spawn(DESTRUCTOR, destructor_locations)

            # cost_upgrade_destructor = game_state.type_cost(DESTRUCTOR, upgrade=True)
            # cost_upgrade_filter = game_state.type_cost(FILTER, upgrade=True)

            # target_cost_upgrade = 0.5 * float(game_state.get_resource(CORES))

            # l = int(len(self.damage_priority)/2)
            # upgrade_priority = heapq.nsmallest(l,self.damage_priority)

            # for (k,v) in upgrade_priority:
            #     if target_cost_upgrade < 0.:
            #         break
            #     if k < 0: # more negative means more damage soaked
            #         units = game_state.game_map[[v[0],v[1]]]
            #         for u in units:
            #             if u is not None and u.player_index == 0 and gamelib.is_stationary(u.unit_type) and not u.upgraded:
            #                 c = game_state.type_cost(u.unit_type, upgrade=True)[CORES]
            #                 target_cost_upgrade = target_cost_upgrade - float(c)
            #                 game_state.attempt_upgrade([v[0],v[1]])

        def plan_attack(game_state, passes, plan):
            
            if game_state.get_resource(BITS) > 15:
                plan["can_attack"] = True
            else:
                plan["can_attack"] = False

            gamelib.debug_write("can attack: {}".format(plan["can_attack"]))

        def build_attack(game_state, passes, plan):

            locs_info_spawn = create_line(start=[6,7],end=[7,6])
            # locs_info_spawn = create_line(start=[13,0],end=[14,0])

            # if game_state.get_resource(BITS) > 1:
            #     num_ping = int(max(0.1*float(game_state.number_affordable(SCRAMBLER)),1.))
            #     sel = random.choice(locs_info_spawn)
            #     game_state.attempt_spawn(SCRAMBLER, sel, num_ping)
            
            # gamelib.debug_write("can attack: {}".format(plan["can_attack"]))
            # gamelib.debug_write("currently_reserving_attacking: {}".format(plan["currently_reserving_attacking"]))

            if game_state.get_resource(BITS) > 15 and plan["currently_reserving_attacking"] == False and plan["can_attack"] == True:

                gamelib.debug_write("deploying attack")
                gamelib.debug_write("can attack: {}".format(plan["can_attack"]))
                gamelib.debug_write("currently_reserving_attacking: {}".format(plan["currently_reserving_attacking"]))
                r = random.random()
                if r > 0.5:
                    num_ping = game_state.number_affordable(PING)
                    sel = random.choice(locs_info_spawn)
                    game_state.attempt_spawn(PING, sel, num_ping)
                else:
                    num_emp = game_state.number_affordable(EMP)
                    sel = random.choice(locs_info_spawn)
                    game_state.attempt_spawn(EMP, sel, num_emp)

            if plan["currently_reserving_attacking"] is True:
                plan["currently_reserving_attacking"] = False

        def build_defense_random(game_state, passes, plan):
            
            destructor_locations = []

            cost_destructor = game_state.type_cost(DESTRUCTOR)[CORES]

            num = int((0.85 * float(game_state.get_resource(CORES))) / float(cost_destructor))
            
            # # l = int(len(self.damage_priority)/2)
            # upgrade_priority = heapq.nsmallest(int(num/2),self.damage_priority)[:2]

            # upgrade_priority = map(lambda a: [a,a,a], upgrade_priority)
            
            # import itertools
            # upgrade_priority = list(itertools.chain.from_iterable(upgrade_priority))
            
            # choices = random.choices(upgrade_priority, k=num)
            # gamelib.debug_write("choices: {}".format(choices))

            if num > 0:

                gmm = create_gaussian_distr(self.damage_map)

                if gmm is not None:
                    j = 0
                    retry = 20
                    while j < num:
                        samp = generate_sample_from_gaussian(gmm, samples=1)
                        x=int(samp[0][0])
                        y=int(samp[0][1])
                        if (x,y) not in plan["attack_path_reserve"]:
                            placement = [[x,y]]
                            # gamelib.debug_write("attempt placement: {}".format(placement))
                            num_spawned, _ = game_state.attempt_spawn(DESTRUCTOR, placement)
                            r = random.random()
                            if num_spawned < 1:
                                retry -= 1
                            else:
                                if r > 0.5:
                                    game_state.attempt_upgrade(placement)
                                j += 1
                                retry = 20
                        else:
                            retry -= 1
                        if retry < 0:
                            j += 1
                            retry = 20

            # todo: upgrades
            # cost_upgrade_destructor = game_state.type_cost(DESTRUCTOR, upgrade=True)
            # cost_upgrade_filter = game_state.type_cost(FILTER, upgrade=True)

            # target_cost_upgrade = 0.5 * float(game_state.get_resource(CORES))

            # l = int(len(self.damage_priority)/2)
            # upgrade_priority = heapq.nsmallest(l,self.damage_priority)

            # for (k,v) in upgrade_priority:
            #     if target_cost_upgrade < 0.:
            #         break
            #     if k < 0: # more negative means more damage soaked
            #         units = game_state.game_map[[v[0],v[1]]]
            #         for u in units:
            #             if u is not None and u.player_index == 0 and gamelib.is_stationary(u.unit_type) and not u.upgraded:
            #                 c = game_state.type_cost(u.unit_type, upgrade=True)[CORES]
            #                 target_cost_upgrade = target_cost_upgrade - float(c)
            #                 game_state.attempt_upgrade([v[0],v[1]])

            # encrytor_locations = create_line(start=[12,1],end=[12,1])
            # encrytor_locations = create_line(start=[7,8],end=[13,2])
            if game_state.get_resource(CORES) > 20:
                encrytor_locations = []
                encrytor_locations.extend(create_line(start=[6,8],end=[6,8]))
                # encrytor_locations.extend(create_line(start=[7,8],end=[12,3]))
                encrytor_locations.extend(create_line(start=[7,8],end=[13,2]))
                game_state.attempt_spawn(ENCRYPTOR, encrytor_locations)

        def reserve_attack_path(game_state, passes, plan):

            if plan["currently_reserving_attacking"] is False and plan["can_attack"] is False:
                plan["attack_path_reserve"] = {}

            locs = []

            # locs.extend(create_line(start=[2,11],end=[21,11]))
            # locs.extend(create_line(start=[6,7],end=[13,0]))
            # locs.extend(create_line(start=[7,7],end=[14,0]))

            # locs.extend(create_line(start=[14,0],end=[14,9]))
            
            # locs.extend(create_line(start=[14,9],end=[21,9]))
            # locs.extend(create_line(start=[21,9],end=[21,13]))

            locs.extend(create_line(start=[2,11],end=[25,11]))
            locs.extend(create_line(start=[13,0],end=[13,11]))
            locs.extend(create_line(start=[14,0],end=[14,11]))
            
            for i in locs:
                plan["attack_path_reserve"][(i[0],i[1])] = i

        def reserve_attack_path_2(game_state, passes, plan):

            if plan["can_attack"] == True and "attack_path_reserve_x_coord" in plan:

                x_coord = plan["attack_path_reserve_x_coord"]
                locs = []

                locs.extend(create_line(start=[x_coord,11],end=[x_coord,13]))

                for i in locs:

                    plan["attack_path_reserve"][(i[0],i[1])] = i
                    gamelib.debug_write("reserve for attack: {}".format(i))
                                    
                    if game_state.contains_stationary_unit(i):
                        for unit in game_state.game_map[i]:
                            if unit is not None and unit.player_index == 0 and gamelib.is_stationary(unit.unit_type):
                                
                                plan["currently_reserving_attacking"] = True

                                #remove structures in path
                                game_state.attempt_remove(i)
        
        #test reserve path for attack here
        # plan["attack_path_reserve_x_coord"] = 25
        
        passes.append(plan_attack)
        passes.append(reserve_attack_path)
        passes.append(reserve_attack_path_2)
        passes.append(build_defense_basic)
        passes.append(build_attack)
        passes.append(build_defense_random)

        # plan["currently_reserving_attacking"] = False

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
