# * AI TAK Bot
# * Sri Ram Bandi (srirambandi.654@gmail.com)

import sys
import pdb
import time
import random
from copy import deepcopy
from collections import deque
from math import exp

class Game:

    class Player:

        def __init__(self, flats, capstones):
            self.flats = flats
            self.capstones = capstones

    def __init__(self, n):
        self.n = n
        self.total_squares = n * n
        self.board = [ [] for i in xrange(self.total_squares) ]
        self.turn = 0
        if n == 5:
            self.max_flats = 21
            self.max_capstones = 1
        elif n == 6:
            self.max_flats = 30
            self.max_capstones = 1
        elif n == 7:
            self.max_flats = 40
            self.max_capstones = 1
        self.max_movable = n
        self.max_down = 1
        self.max_up = n
        self.max_left = 'a'
        self.max_right = chr(ord('a') + n - 1)
        self.moves = 0
        self.players = []
        self.players.append(Game.Player(self.max_flats, self.max_capstones))
        self.players.append(Game.Player(self.max_flats, self.max_capstones))
        self.all_squares = [ self.square_to_string(i) for i in xrange(self.total_squares) ]

    def square_to_num(self, square_string):
        """ Return -1 if square_string is invalid
        """
        if len(square_string) != 2:
            return -1
        if not square_string[0].isalpha() or not square_string[0].islower() or not square_string[1].isdigit():
            return -1
        row = ord(square_string[0]) - 96
        col = int(square_string[1])
        if row < 1 or row > self.n or col < 1 or col > self.n:
            return -1
        return self.n * (col - 1) + (row - 1)

    def square_to_string(self, square):
        """Convert square number to string
        """
        if square < 0 or square >= self.total_squares:
            return ''
        row = square % self.n
        col = square / self.n
        return chr(row + 97) + str(col + 1)

    def unexecute_move(self, move_string):
        """Unexecute placement move
        """
        if self.moves == 1 or self.moves == 0:
            current_piece = 1 - self.turn
        else:
            current_piece = self.turn
        square = self.square_to_num(move_string[1:])
        self.board[square].pop()
        if move_string[0] == 'C':
            self.players[current_piece].capstones += 1
        else:
            self.players[current_piece].flats += 1

    def execute_move(self, move_string):
        """Execute move
        """
        if self.turn == 0:
            self.moves += 1
        if self.moves != 1:
            current_piece = self.turn
        else:
            current_piece = 1 - self.turn
        if move_string[0].isalpha():
            square = self.square_to_num(move_string[1:])
            if move_string[0] == 'F' or move_string[0] == 'S':
                self.board[square].append((current_piece, move_string[0]))
                self.players[current_piece].flats -= 1
            elif move_string[0] == 'C':
                self.board[square].append((current_piece, move_string[0]))
                self.players[current_piece].capstones -= 1
        elif move_string[0].isdigit():
            count = int(move_string[0])
            square = self.square_to_num(move_string[1:3])
            direction = move_string[3]
            if direction == '+':
                change = self.n
            elif direction == '-':
                change = -self.n
            elif direction == '>':
                change = 1
            elif direction == '<':
                change = -1
            prev_square = square
            for i in xrange(4, len(move_string)):
                next_count = int(move_string[i])
                next_square = prev_square + change
                if len(self.board[next_square]) > 0 and self.board[next_square][-1][1] == 'S':
                    self.board[next_square][-1] = (self.board[next_square][-1][0], 'F')
                if next_count - count == 0:
                    self.board[next_square] += self.board[square][-count:]
                else:
                    self.board[next_square] += self.board[square][-count:-count + next_count]
                prev_square = next_square
                count -= next_count

            count = int(move_string[0])
            self.board[square] = self.board[square][:-count]
        self.turn = 1 - self.turn

    def partition(self, n):
        """Generates all permutations of all partitions
        of n
        """
        part_list = []
        part_list.append([n])
        for x in xrange(1, n):
            for y in self.partition(n - x):
                part_list.append([x] + y)

        return part_list

    def check_valid(self, square, direction, partition):
        """For given movement (partition), check if stack on
        square can be moved in direction. Assumes active player
        is topmost color
        """
        if direction == '+':
            change = self.n
        elif direction == '-':
            change = -self.n
        elif direction == '>':
            change = 1
        elif direction == '<':
            change = -1
        for i in xrange(len(partition)):
            next_square = square + change * (i + 1)
            if len(self.board[next_square]) > 0 and self.board[next_square][-1][1] == 'C':
                return False
            if len(self.board[next_square]) > 0 and self.board[next_square][-1][1] == 'S' and i != len(partition) - 1:
                return False
            if i == len(partition) - 1 and len(self.board[next_square]) > 0 and self.board[next_square][-1][1] == 'S' and partition[i] > 1:
                return False
            if i == len(partition) - 1 and len(self.board[next_square]) > 0 and self.board[next_square][-1][1] == 'S' and self.board[square][-1][1] != 'C':
                return False

        return True

    def generate_stack_moves(self, square):
        """Generate stack moves from square
        Assumes active player is topmost color
        """
        all_moves = []
        r = square % self.n
        c = square / self.n
        size = len(self.board[square])
        dirs = ['+',
         '-',
         '<',
         '>']
        up = self.n - 1 - c
        down = c
        right = self.n - 1 - r
        left = r
        rem_squares = [up,
         down,
         left,
         right]
        for num in xrange(min(size, self.n)):
            part_list = self.partition(num + 1)
            for di in range(4):
                part_dir = [ part for part in part_list if len(part) <= rem_squares[di] ]
                for part in part_dir:
                    if self.check_valid(square, dirs[di], part):
                        part_string = ''.join([ str(i) for i in part ])
                        all_moves.append(str(sum(part)) + self.all_squares[square] + dirs[di] + part_string)

        return all_moves

    def generate_all_moves(self, player):
        """Generate all possible moves for player
        Returns a list of move strings
        """
        all_moves = []
        for i in xrange(len(self.board)):
            if len(self.board[i]) == 0:
                if self.players[player].flats > 0:
                    all_moves.append('F' + self.all_squares[i])
                if self.moves != player and self.players[player].flats > 0:
                    all_moves.append('S' + self.all_squares[i])
                if self.moves != player and self.players[player].capstones > 0:
                    all_moves.append('C' + self.all_squares[i])

        for i in xrange(len(self.board)):
            if len(self.board[i]) > 0 and self.board[i][-1][0] == player and self.moves != player:
                all_moves += self.generate_stack_moves(i)

        return all_moves


class Agent:

    def __init__(self):
        data = sys.stdin.readline().strip().split()
        self.player = int(data[0]) - 1
        self.n = int(data[1])
        self.time_left = int(data[2])
        self.game = Game(self.n)
        self.max_depth = self.n
        self.play()

    def getNeighbors(self, square,length=1):
		total_squares = self.n * self.n
		if square < 0 or square >= total_squares:
			return []
		elif square == 0:
			arr = []
			for i in xrange(length):
				arr.append(square + 1+i)
				arr.append(square + (1+i)*self.n)
			return arr
		elif square == self.n - 1:
			arr = []
			for i in xrange(length):
				arr.append(square - 1-i)
				arr.append(square + (1+i)*self.n)
			return arr
		elif square == total_squares - self.n:
			arr = []
			for i in xrange(length):
				arr.append(square + 1+i)
				arr.append(square - (1+i)*self.n)
			return arr
		elif square == total_squares - 1:
			arr = []
			for i in xrange(length):
				arr.append(square - 1-i)
				arr.append(square - (1+i)*self.n)
			return arr
		elif square < self.n:
			arr = []
			for i in xrange(length):
				if square - 1-i >= 0:
					arr.append(square - 1-i)
				if square + 1+i < self.n:
					arr.append(square + 1+i)
				arr.append(square + (1+i)*self.n)
			return arr
		elif square % self.n == 0:
			arr = []
			for i in xrange(length):
				arr.append(square + 1+i)
				if square + (1+i)*self.n < total_squares:
					arr.append(square + (1+i)*self.n)
				if square - (1+i)*self.n >=0 :
					arr.append(square - (1+i)*self.n)
			return arr
		elif (square + 1) % self.n == 0:
			arr = []
			for i in xrange(length):
				arr.append(square - 1-i)
				if square + (1+i)*self.n < total_squares:
					arr.append(square + (1+i)*self.n)
				if square - (1+i)*self.n >= self.n-1 :
					arr.append(square - (1+i)*self.n)
			# arr = [elm if 0<= elm <self.n for elm in arr]
			return arr
		elif square >= total_squares - self.n:
			arr = []
			for i in xrange(length):
				if square - 1-i >= total_squares-self.n:
					arr.append(square - 1-i)
				if square + 1+i < total_squares:
					arr.append(square + 1+i)
				arr.append(square - (1+i)*self.n)
			# arr = [elm if 0<= elm <self.n for elm in arr]
			return arr
		else:
			arr = []
			for i in xrange(length):
				if square + 1+i < ((square/self.n)+1)*self.n:
					arr.append(square + 1+i)
				if square - 1-i >= ((square/self.n))*self.n:
					arr.append(square - 1-i)
				if square + (1+i)*self.n <= total_squares-self.n +(square%self.n):
					arr.append(square + (1+i)*self.n)
				if square - (1+i)*self.n >= (square%self.n):
					arr.append(square - (1+i)*self.n)
			return arr

    def bfs(self, source, direction, player):
        if direction == '<' and source % self.n == 0 or direction == '-' and source // self.n == 0 or direction == '>' and (source + 1) % self.n == 0 or direction == '+' and source // self.n == self.n - 1:
            return 0
        fringe = deque()
        fringe.append((source, 0))
        value = 0
        reached_end = False
        while not len(fringe) == 0 and value < self.n + 1:
            node, val = fringe.popleft()
            value = val
            if direction == '<' and node % self.n == 0 or direction == '-' and node // self.n == 0 or direction == '>' and (node + 1) % self.n == 0 or direction == '+' and node // self.n == self.n - 1:
                reached_end = True
                break
            nbrs = self.getNeighbors(node)
            for nbr in nbrs:
                if len(self.game.board[nbr]) == 0 or self.game.board[nbr][-1][0] == player and self.game.board[nbr][-1][1] != 'S':
                    if direction == '<':
                        if nbr % self.n <= node % self.n:
                            fringe.append((nbr, value + 1))
                    elif direction == '-':
                        if nbr // self.n <= node // self.n:
                            fringe.append((nbr, value + 1))
                    elif direction == '>':
                        if nbr % self.n >= node % self.n:
                            fringe.append((nbr, value + 1))
                    elif direction == '+':
                        if nbr // self.n >= node // self.n:
                            fringe.append((nbr, value + 1))

        if reached_end:
            return value
        else:
            return self.n * self.n

    def dfs(self, source, direction, player, visited):
        fringe = []
        fringe.append(source)
        dfs_val = self.n * self.n
        while not len(fringe) == 0:
            node = fringe.pop()
            visited.add(node)
            nbrs = self.getNeighbors(node)
            has_children = False
            for nbr in nbrs:
                if len(self.game.board[nbr]) > 0 and nbr not in visited and self.game.board[nbr][-1][0] == player and self.game.board[nbr][-1][1] != 'S':
                    fringe.append(nbr)
                    has_children = True

            if not has_children:
                if direction == '>':
                    dfs_val = min(self.bfs(node, '>', player), dfs_val)
                elif direction == '+':
                    dfs_val = min(self.bfs(node, '+', player), dfs_val)

        if dfs_val == self.n * self.n:
            return -1
        else:
            if dfs_val != 0:
                pass
            return dfs_val

    def score_combination(self, bfs, dfs):
        if dfs == -1:
            return 0
        else:
            return self.n * self.n * exp(-0.5 * (bfs + dfs))

    def road_score(self, player):

        value = 0
        visited = [set(), set()]
        for r in xrange(0, self.n / 2):
            for c in xrange(0, self.n):
                idx = r * self.n + c
                if len(self.game.board[idx]) > 0 and self.game.board[idx][-1][0] == player and self.game.board[idx][-1][1] != 'S' and idx not in visited[1]:
                    value = max(value, self.score_combination(self.bfs(idx, '-', player), self.dfs(idx, '+', player, visited[1])))
                idx = c * self.n + r
                if len(self.game.board[idx]) > 0 and self.game.board[idx][-1][0] == player and self.game.board[idx][-1][1] != 'S' and idx not in visited[0]:
                    value = max(value, self.score_combination(self.bfs(idx, '<', player), self.dfs(idx, '>', player, visited[0])))

        return value

    def evaluation_function(self):
 		if self.n == 5:
			bonus = 1000;
			cutoff_bonus = 20000;
			fc = 100;
			cs = 85;
			w = 30;
			c = 15;
			ccap = 15;
			s = 18;
			inf = 10;
			rd = 10;
			wallpenalty = 10;
			attack = 1;
			fw = 15.5;
			ccw = 6.0;
			sw = 3.0;
			iw = 6.0;
			rw = 4.0;
			cw = 4.0;
			ww = 1.0;
			cow = 1.3;
		else:
			bonus = 2500;
			cutoff_bonus = 45000;
			fc = 100;
			cs = 85;
			w = 40;
			c = 15;
			ccap = 20;
			s = 18;
			rd = 1;
			inf = 10;
			wallpenalty = 10;
			attack = 2;
			fw = 15.5;
			ccw = 6.0;
			sw = 3.0;
			iw = 6.0;
			rw = 4.0;
			cw = 4.0;
			ww = 1;
			cow = 1.3;

		infl = [[0,0,0,0,0,0,0],
				[0,1,0,1,1,0,0],
				[0,0,0,0,1,0,0],
				[0,1,1,1,1,1,0],
				[0,-1,0,0,0,0,-1],
				[0,-1,0,0,-1,0,0],
				[0,-1,-1,-1,-1,-1,0]]
		flats_owned = [0, 0]
		color = [0,0]
		stack_score = 0;
		wall_dis = 0;
		total_influence =0;
		composition = 0;
		total_squares = len(self.game.board)
		influence_table = [0 for i in xrange(total_squares)]
		for idx in xrange(len(self.game.board)):

			################# Feature 1 #################
			if len(self.game.board[idx]) > 0:
				if self.game.board[idx][-1][1] == 'F':
					flats_owned[self.game.board[idx][-1][0]] += fc
				if self.game.board[idx][-1][1] == 'C':
					flats_owned[self.game.board[idx][-1][0]] += cs
				if self.game.board[idx][-1][1] == 'S':
					flats_owned[self.game.board[idx][-1][0]] += w

			##############################################
			################# Feature 2 #################
			if len(self.game.board[idx]) > 1: #if it is stack
				for h in xrange(len(self.game.board[idx])-1):
					color[self.game.board[idx][h][0]] += s

				stack_score = color[self.player] - color[1 - self.player]

				piece = self.game.board[idx][-1][1];
				owner = self.game.board[idx][-1][0];


				if owner == self.player and stack_score >0:
					if piece == 'F':
						stack_score += 4*s;
					if piece == 'C':
						stack_score += 7*s;
					else:
						stack_score += 5*s;
				elif owner == self.player and stack_score <=0:
					if piece == 'F':
						stack_score += -2*s;
					if piece == 'C':
						stack_score += -5*s;
					else:
						stack_score += -3*s;
				elif owner != self.player and stack_score >0:
					if piece == 'F':
						stack_score += 2*s;
					if piece == 'C':
						stack_score += 5*s;
					else:
						stack_score += 3*s;
				else:
					if piece == 'F':
						stack_score += -4*s;
					if piece == 'C':
						stack_score += -7*s;
					else:
						stack_score += -5*s;

				if owner == self.player and color[self.player]>0:
					ttt = min(self.n-1,color[self.player]+1)
					nbrs = nbrs = self.getNeighbors(idx,ttt)
					for nbr in nbrs:
						if len(self.game.board[nbr]) == 0 :
							influence_table[nbr] += 1;
						elif  self.game.board[nbr][-1] == [1-self.player,'F']:
							influence_table[nbr] += 1;
						elif self.game.board[nbr][-1][1] != 'F':
							influence_table[nbr] -= 1;
						elif piece == 'C' and self.game.board[idx][-2][0] == self.player and self.game.board[nbr][-1] == [1-self.player,'S']:
							influence_table[nbr] += 1;

						if abs(idx-nbr)==1 :
							if len(self.game.board[nbr]) > 0 :
								if self.game.board[nbr][-1][0] == self.player and self.game.board[nbr][-1][1] != 'F':
									wall_dis -= color[self.player]*wallpenalty;
								if self.game.board[nbr][-1][0] == 1-self.player and self.game.board[nbr][-1][1] != 'F':
									wall_dis -= color[1-self.player]*wallpenalty
			##############################################

			##################Feature 3 ##################
			if len(self.game.board[idx]) == 1:
				nbrs = nbrs = self.getNeighbors(idx)
				mapping = {'F':1,'S':2,'C':3}
				piece = self.game.board[idx][-1][1];
				owner = self.game.board[idx][-1][0];
				for nbr in nbrs:
					if len(self.game.board[nbr]) > 0:
						nbr_piece = self.game.board[nbr][-1][1]
						nbr_owner = self.game.board[nbr][-1][0]
						xx = mapping[piece] if owner==self.player else 3+mapping[piece]
						yy = mapping[nbr_piece] if nbr_owner==self.player else 3+mapping[nbr_piece]
						influence_table[nbr] += infl[xx][yy]

		row = 0;
		for i in range(0,total_squares,self.n):
			row_num = 0;
			for j in range(i,i+self.n):
				if len(self.game.board[j]) > 0:
					if self.game.board[j][-1][0]== self.player and self.game.board[j][-1][1] != 'S':
						row_num +=1;
					elif self.game.board[j][-1][0]== 1-self.player and self.game.board[j][-1][1] != 'S':
						row_num -= attack;
				composition += row_num**3;
				total_influence += influence_table[j]*10;
				row += influence_table[j]*10

		col = 0;
		for i in range(self.n):
			col_num = 0;
			for _j in range(self.n):
				j = i+ (_j*self.n);
				if len(self.game.board[j]) > 0:
					if self.game.board[j][-1][0]== self.player and self.game.board[j][-1][1] != 'S':
						col_num +=1;
					elif self.game.board[j][-1][0]== 1-self.player and self.game.board[j][-1][1] != 'S':
						col_num -= attack;
				composition += col_num**3;
				# total_influence += influence_table[j]*10;
				col += influence_table[j]*10
		flat_comp = flats_owned[self.player] - flats_owned[1 - self.player]
		road_comp =  self.road_score(1 - self.player) + self.road_score(self.player)
		return rd*road_comp + fw*flat_comp  + sw*stack_score + iw*total_influence + rw* row + cw*col + ww*wall_dis + cow*composition;
    def play(self):
    	a = random.randint(0,self.n-1)
    	b = random.randint(1,self.n)
        if self.player == 0:
			move = 'F' + chr(97+a)+str(b)
			self.game.execute_move(move)
			sys.stdout.write(move + '\n');
			sys.stdout.flush()

			opponent_move = sys.stdin.readline().strip()

			move = self.min_max(opponent_move)
			sys.stdout.write(move + '\n');
			sys.stdout.flush()


        else:
			opponent_move = sys.stdin.readline().strip()
			self.game.execute_move(opponent_move)

			move = 'F' + chr(97+a)+str(b)
			while(move == opponent_move):
				a = random.randint(0,self.n-1)
				b = random.randint(1,self.n)
				move = 'F' + chr(97+a)+str(b)

			self.game.execute_move(move)
			sys.stdout.write(move + '\n');
			sys.stdout.flush()

        while True:
            opponent_move = sys.stdin.readline().strip()

            move = self.min_max(opponent_move)

            sys.stdout.write(move + '\n')
            sys.stdout.flush()

    def min_max(self,move):
    	if move != '':
    		self.game.execute_move(move)
        (my_move,val) = self.max_node(float('-inf'), float('inf'), 2,move)
        self.game.execute_move(my_move)
        return my_move

    def unexecute_move(self, board, players, moves, turn):
        self.game.board = board
        self.game.players = players
        self.game.moves = moves
        self.game.turn = turn

    def max_node(self, alpha, beta, depth,opponent_move):
        if depth == 0:
            return (opponent_move,self.evaluation_function())
        val = float('-inf')
        # store_val = False
        # if depth == 0:
        #     store_val = True
        action = opponent_move;
        for move in self.game.generate_all_moves(self.player):
            old_game = deepcopy(self.game)
            self.game.execute_move(move)
            node_val = self.min_node(alpha, beta, depth - 1)
            self.game = old_game
            val = max(node_val, val)
            if val == node_val:
                action = move
            alpha = max(alpha, val)
            if beta <= alpha:
                break

        # if store_val:
        #     sys.stderr.write('Value: ' + str(val) + '\n')
        #     return (action,val)
        # else:
        #     return (action,val)
        return (action,val)

    def min_node(self, alpha, beta, depth):
        if depth == 0:
            return self.evaluation_function()
        val = float('inf')
        # store_val = False
        # if depth == 0:
        #     store_val = True
        action = ''
        for move in self.game.generate_all_moves(1 - self.player):
            old_game = deepcopy(self.game)
            self.game.execute_move(move)
            (some_move,node_val) = self.max_node(alpha, beta, depth - 1,move)
            self.game = old_game
            val = min(node_val, val)
            if val == node_val:
                action = move
            beta = min(beta, val)
            if beta <= alpha:
                break

        # if store_val:
        #     sys.stderr.write('Value: ' + str(val) + '\n')
        #     return action
        # else:
        #     return val
        return val


random_player = Agent()
# +++ okay decompyling AgentBFS.pyc
# decompiled 1 files: 1 okay, 0 failed, 0 verify failed
# 2016.10.20 22:47:40 IST
