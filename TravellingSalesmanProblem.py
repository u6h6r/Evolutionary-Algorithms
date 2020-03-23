#! /usr/bin/env python3
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from numpy.random import choice
from random import choices
import numpy as np
import itertools
import operator
import random
import copy

class Select: 
  current_population={}
  crossover_population={}
  route_values={}
  itaretion=0
  parent=0
  shortest_route=10000
  
  iteration_plot=[]
  shortest_route_plot=[]
  
def random_route(): #Create random route for selected chromosome
    route=random.sample(city_data,len(city_data))
    return route                         

def initial_population(p,city_data): #Create initial population of chromosome in format dict{key[unique_number]:chromosome} 
    p_permutations={}
    for p in range(0,p):
      p_permutations[p]=random_route()  
    return p_permutations

def total_distance(route): #Calculates route between all cities 
    distance=0
    x_distance=[]
    y_distance=[]

    for i in range(0,len(route)-1):
        x_distance.append(abs(route[i][0] - route[i+1][0]))
        y_distance.append(abs(route[i][1] - route[i+1][1]))
        distance+=np.sqrt(pow(x_distance[i],2)+pow(y_distance[i],2))

    distance+=np.sqrt(pow((route[0][0]-route[len(route)-1][0]),2)+pow((route[0][1]-route[len(route)-1][1]),2)) 

    return distance

def inverse_probability(probability_list): #Invert weights of routes for roulette selection.
  weights = [1.0 / w for w in probability_list]           
  sum_weights = sum(weights)
  weights = [w / sum_weights for w in weights]
  return weights

def selection_operator(permutation,crossover_size): #Selection operator for crossover via roulette wheel
  
  Select.current_population=permutation.copy()

  dictionary_of_probabilities={}

  for p in permutation:
    dictionary_of_probabilities[p]=total_distance(permutation.get(p)) #Format dict{key[unique_number]:route distance}

  population_size=len(Select.current_population) #Size of parents population

  for parent in list(Select.current_population):   #Iteration over elements of permutation
    if (len(Select.crossover_population)/population_size)<crossover_size: #Crossover size    
      population_distance = sum(list(dictionary_of_probabilities.values())) #Calculates whole route of salesman
      probability=[]                         
      for i in list(dictionary_of_probabilities.values()): #List of probabilities
        probability.append(i/population_distance)   
      invers_prob = inverse_probability(probability) #Inverts probability, shortest road got highest probability

      Select.parent=choice(list(Select.current_population.keys()),1,p=invers_prob)
      Select.parent=int(Select.parent[0])#Becouse returned parent is type: list and this changing type to int

      Select.crossover_population[Select.parent]=Select.current_population.get(Select.parent)#Format dict{key[unique_number]:chromosome}
      
      dictionary_of_probabilities.pop(Select.parent,None)
      Select.current_population.pop(Select.parent,None)

def empty_offspring(lenght):
  off_1 = [None for i in range(lenght)]
  off_2 = [None for i in range(lenght)]
  return off_1, off_2

def cycle_crossover(off, p1, p2):
  current_index = 0
  new_off = list(off)
  new_off[0] = p1[0]
  # print ("Actual new_off: {} current_index: {}".format(new_off, current_index))
  while None in new_off:
      next_allele_2 = p2[current_index]
      if next_allele_2 not in new_off:
          current_index = p1.index(next_allele_2)
          new_off[current_index] = next_allele_2
          # print ("Actual new_off: {} current_index: {}".format(new_off, current_index))
      else:
          # print ("Actual new_off: {} current_index: {}".format(new_off, current_index))
          final_off = fill_offspring(new_off, p2)
          # print ("Final off: {}".format(final_off))
          return final_off
  return new_off
                    
def fill_offspring(off, p2):
  fill_off = list(off)
  for i in range(len(fill_off)):
      if fill_off[i] is None:
          fill_off[i] = p2[i]
  return fill_off

def cx_operator(parent1,parent2):
  p1 = list(parent1)
  p2 = list(parent2)
  lenght = len(p1)
  off_1, off_2 = empty_offspring(lenght)
  new_off = cycle_crossover(off_1, p1, p2)
  return new_off

def mutate(off, pm): # wymaga
  new_off = list(off)
  # print ("Off before mutation:\t {}".format(new_off))
  idxs = range(len(off))
  # print ("indx:\t\t\t {}".format(idxs))
  n = len(off)
  # print ("n: {}".format(n))
  k = int(2*(n*pm))
  # print ("k: {}".format(k))
  indx_list = random.sample(idxs, k)
  # print ("index list:\t\t {}".format(indx_list))
  mutated_off = swap(new_off, indx_list)
  # print ("Off after mutation:\t {}".format(mutated_off))
  return mutated_off

def swap(off, indx_list):
  swap_tuples = []
  new_off = list(off)
  count = 0
  swaps = []
  for pos in indx_list:
      swaps.append(pos)
      count += 1
      if count == 2:
          swap_tuples.append(swaps)
          count = 0
          swaps = []
  # print ("Swap tuples:\t\t {}".format(swap_tuples))
  for swp in swap_tuples:
      new_off[swp[0]], new_off[swp[1]] = new_off[swp[1]], new_off[swp[0]]
  return new_off

def offspring_generator(pm): #Generates two offspring by 2 parents   !!! MOZNA TO JESZCZE JAKOS SKROCIC!
  
  parents={}

  for i in range(2):
    parents[i]=[random.choice(list(Select.crossover_population.keys()))] #parent key : random number,chromosome : parents[0]= [cyfra]
    parents[i].append(Select.crossover_population.get(parents[i][0]))
    Select.crossover_population.pop(parents[i][0],None)

  offspring_1=cx_operator(parents[0][1],parents[1][1])
  offspring_2=cx_operator(parents[1][1],parents[0][1])

  mutated_offspring_1=mutate(offspring_1, pm)
  mutated_offspring_2=mutate(offspring_2, pm)

  Select.crossover_population[parents[0][0]]=mutated_offspring_1
  Select.crossover_population[parents[1][0]]=mutated_offspring_2

def genetic_algorithm(city_data,P,n,pm,T):
  
  Select.current_population=initial_population(P,city_data) 
  
  crossover_size=int(((len(Select.current_population)*n)/2))

  Select.route_values={}
  
  while Select.itaretion<T:

      selection_operator(Select.current_population,n)

      for i in range(crossover_size):
            
        offspring_generator(pm)

      Select.current_population.update(Select.crossover_population) 
      Select.crossover_population={}
      
            ### Poprawki zwiazane z lepsza wydajnoscia znalezienia minimum 

      for i in Select.current_population:
          Select.route_values[round((total_distance(Select.current_population.get(i))),3)]=Select.current_population.get(i)

      minimum=min(Select.route_values.keys())

      if minimum<Select.shortest_route:
        Select.shortest_route=minimum
      
    # print("Shortest possible route in this iteration is {} and the chromsome is {}".format(Select.shortest_route,Select.route_values[Select.shortest_route]))
      
      Select.iteration_plot.append(Select.itaretion)
      Select.shortest_route_plot.append(Select.shortest_route)

      Select.itaretion+=1
  
  print("Shortest possible route to considered traveling salesman problem is {}".format(Select.shortest_route))

  draw_plot()
 
def draw_plot():
      
  X1,Y1 = zip(*city_data)
  X,Y = zip(*Select.route_values[Select.shortest_route])

  gs = gridspec.GridSpec(2, 2)

  fig = plt.figure()
  ax1 = fig.add_subplot(gs[0, 0]) # row 0, col 0
  for i in range(0,len(city_data)):
    ax1.plot(X1[i:i+2], Y1[i:i+2],'bo-')
  plt.xlabel("Route of traveling salesman before GA",axes=ax1)

  ax2 = fig.add_subplot(gs[0, 1]) # row 0, col 1
  for i in range(0,len(Select.route_values[Select.shortest_route])):
      ax2.plot(X[i:i+2], Y[i:i+2],'ro-')
  plt.xlabel("Route of traveling salesman after GA",axes=ax2)

  ax3 = fig.add_subplot(gs[1, :]) # row 1, span all columns
  ax3.plot(Select.iteration_plot,Select.shortest_route_plot,'k-')
  plt.ylabel("Route distance",axes=ax3)
  plt.xlabel("Algorithm iteration",axes=ax3)
  plt.show()

  pass

if __name__ == "__main__":

    city_data=[(0,1),(3,2),(6,1),(7,4.5),(15,-1),(10,2.5),(16,11),(5,6),(8,9),(1.5,12)] #Coordinates of city map birthdate 30.05.1996r. 
                                                                                        #(3+0+0+5+1+9+9+6)%5=3

    genetic_algorithm(city_data,250,0.8,0.2,100)
  

    
