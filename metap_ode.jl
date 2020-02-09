
#####################################################################################################################
################################# Preface: a practical example of METAPROGRAMMING ###################################
#####################################################################################################################

# A key question in systems biology is to understand what network topologies are capable of mediating particular biological functions.
# Studies as such include:
# "Defining Network Topologies that Can Achieve Biochemical Adaptation" by Ma et al. (2009) Cell 
# "Retroactivity Affects the Adaptive Robustness of Transcriptional Regulatory Networks," by Wang et al. (2019) Proceedings of American Control Conference 
# There are altogether 3^9 = 19683 three-gene network topologies.  It is impossible to code each model manually. 
# Instead, we can automate the simulation by representing the topologies with tertiary numbers and loop through each topology.
# MOREVER, Julia compiles a version of the function where every variable is statically typed when the function is called for the first time.
# Because of this, we can SIGNIFICANTLY accelerate the simulation with metaprogramming!
# This script shows how to use metaprogramming to enumerate and simulate ODE models for all three-gene topologies in Julia


######## load packages ######
using DifferentialEquations, LinearAlgebra, Statistics, Random


##################################################################################
################################# Define Functions ###############################
##################################################################################

######## generating model parameters based on the network topology ########
## n_k: Hill coefficient, 0.5 ~ 4.0
## K_k: dissociation constant, 0.001 ~ 1.0 (log scale)
## tau: decay rate, 0.01 ~ 1.0 (log scale)
## type: type of regulation, parent: index of parent node (1=A, 2=B, 3=C, 4=inducer)

function generate_rand_params(interaction_matrix, InitRandStream=true)
    rand_params_matrix = Dict[]    # a dictionary-type variable that stores all the param values
    for i = 1:size(interaction_matrix, 2)   # for each node (gene) of the network
        rand_params = Dict()
        column = interaction_matrix[:, i]
        if sum(abs.(column)) == 1   # the given node has only one parent 
            if sum(column .== 1) == 1   # activation
                rand_params["type_1"] = "a"  
                rand_params["parent_1"] = findall(x->x==1, column)[1]
            elseif sum(column .== -1) == 1   # inhibition
                rand_params["type_1"] = "r"
                rand_params["parent_1"] = findall(x->x==-1, column)[1]
            end
            n_1 = 0.5+rand()*3.5  
            K_1 = 10.0^(-3+rand()*3)
            tau = 10.0^(rand() * 2)
            rand_params["n_1"] = n_1
            rand_params["K_1"] = K_1
            rand_params["tau"] = 1/tau
        elseif sum(abs.(column)) == 2   # the given node has two parents
            if sum(column .== 1) == 2   # both activation     
            	one_index = findall(x->x==1, column)        
                rand_params["type_1"] = "a"
                rand_params["parent_1"] = one_index[1]
                rand_params["type_2"] = "a"
                rand_params["parent_2"] = one_index[2]
            elseif sum(column .== -1) == 2   # both inhibition    
            	minus_one_index = findall(x->x==-1, column)          
                rand_params["type_1"] = "r"
                rand_params["parent_1"] = minus_one_index[1]
                rand_params["type_2"] = "r"
                rand_params["parent_2"] = minus_one_index[2]
            else   # one activation, one inhibition
                rand_params["type_1"] = "a"
                rand_params["parent_1"] = findall(x->x==1, column)[1]
                rand_params["type_2"] = "r"
                rand_params["parent_2"] = findall(x->x==-1, column)[1]
            end
            n_1 = 0.5+rand()*3.5
            K_1 = 10.0^(-3+rand()*3)
            n_2 = 0.5+rand()*3.5
            K_2 = 10.0^(-3+rand()*3)
            tau = 10.0^(rand()*2)
            rand_params["n_1"] = n_1
            rand_params["K_1"] = K_1
            rand_params["n_2"] = n_2
            rand_params["K_2"] = K_2
            rand_params["tau"] = 1/tau
        elseif sum(abs.(column)) == 3   # the given node has three parents
            if sum(column .== 1) == 0   # all inhibition
                minus_one_index = findall(x->x==-1, column)
                rand_params["type_1"] = "r"
                rand_params["parent_1"] = minus_one_index[1]
                rand_params["type_2"] = "r"
                rand_params["parent_2"] = minus_one_index[2]
                rand_params["type_3"] = "r"
                rand_params["parent_3"] = minus_one_index[3]
            elseif sum(column .== 1) == 1   # one activation, two inhibition
                rand_params["type_1"] = "a"
                rand_params["parent_1"] = findall(x->x==1, column)[1]
                minus_one_index = findall(x->x==-1, column)
                rand_params["type_2"] = "r"
                rand_params["parent_2"] = minus_one_index[1]
                rand_params["type_3"] = "r"
                rand_params["parent_3"] = minus_one_index[2]
            elseif sum(column .== 1) == 2   # two activation, one inhibition
                one_index = findall(x->x==1, column)
                rand_params["type_1"] = "a"
                rand_params["parent_1"] = one_index[1]
                rand_params["type_2"] = "a"
                rand_params["parent_2"] = one_index[2]
                rand_params["type_3"] = "r"
                rand_params["parent_3"] = findall(x->x==-1, column)[1]
            else   # all activation
                one_index = findall(x->x==1, column)
                rand_params["type_1"] = "a"
                rand_params["parent_1"] = one_index[1]
                rand_params["type_2"] = "a"
                rand_params["parent_2"] = one_index[2]
                rand_params["type_3"] = "a"
                rand_params["parent_3"] = one_index[3]
            end
            n_1 = 0.5+rand()*3.5
            K_1 = 10.0^(-3+rand()*3)
            n_2 = 0.5+rand()*3.5
            K_2 = 10.0^(-3+rand()*3)
            n_3 = 0.5+rand()*3.5
            K_3 = 10.0^(-3+rand()*3)              
            tau = 10.0^(rand()*2)
            rand_params["n_1"] = n_1
            rand_params["K_1"] = K_1
            rand_params["n_2"] = n_2
            rand_params["K_2"] = K_2
            rand_params["n_3"] = n_3
            rand_params["K_3"] = K_3
            rand_params["tau"] = 1/tau
        end
        if i == 1   # Special case: within the three nodes, A can be induced.
            if !haskey(rand_params, "type_1")        # if A has no regulator other than the inducer
                rand_params["type_1"] = "i"
                rand_params["parent_1"] = 4
                n_1 = 1.0
                K_1 = 0.4
                tau = 10^(rand()*2)
                rand_params["n_1"] = n_1
                rand_params["K_1"] = K_1
                rand_params["tau"] = 1/tau
            elseif !haskey(rand_params, "type_2")    # if A has one regulator other than the inducer
                rand_params["type_2"] = "i"
                rand_params["parent_2"] = 4
                n_2 = 1.0
                K_2 = 0.4
                rand_params["n_2"] = n_2
                rand_params["K_2"] = K_2
            elseif !haskey(rand_params, "type_3")    # if A has two regulators other than the inducer
                rand_params["type_3"] = "i"
                rand_params["parent_3"] = 4
                n_3 = 1.0
                K_3 = 0.4
                rand_params["n_3"] = n_3
                rand_params["K_3"] = K_3        
            else                                     # if A has three regulators other than the inducer
                rand_params["type_4"] = "i"
                rand_params["parent_4"] = 4
                n_4 = 1.0
                K_4 = 0.4
                rand_params["n_4"] = n_4
                rand_params["K_4"] = K_4
            end
        end
        rand_params["self"] = i                  
        push!(rand_params_matrix, rand_params)
    end
    return rand_params_matrix
end


##### convert parameter dictionary to parameter tuple for faster speed #####

function dict_to_tuples(rand_params_matrix) 
    rand_params_tuple_lst = []
    for rand_params in rand_params_matrix   # for each node (gene) of the topology
        if length(keys(rand_params)) == 6          # the given node has only one parent
            rand_params_tuple = (rand_params["n_1"], rand_params["K_1"],rand_params["tau"])
        elseif length(keys(rand_params)) == 10     # the given node has two parents
            rand_params_tuple = (rand_params["n_1"], rand_params["K_1"], 
            rand_params["n_2"], rand_params["K_2"], rand_params["tau"])
        elseif length(keys(rand_params)) == 14     # the given node has three parents
            rand_params_tuple = (rand_params["n_1"], rand_params["K_1"],
            rand_params["n_2"], rand_params["K_2"], rand_params["n_3"], rand_params["K_3"],
            rand_params["tau"])
        elseif length(keys(rand_params)) == 18     # the given node has 
            rand_params_tuple = (rand_params["n_1"], rand_params["K_1"],
            rand_params["n_2"], rand_params["K_2"], rand_params["n_3"], rand_params["K_3"],
            rand_params["n_4"], rand_params["K_4"], rand_params["tau"])
        end
        push!(rand_params_tuple_lst, rand_params_tuple)
    end
    return rand_params_tuple_lst
end


########## construct the ODE based on the parameters (where metaprogramming takes place) ###########
# input: parameter dictionary
# output: dXdt - change in concentration per unit time
# Rprod: retroactivity matrix

function one_node_ODE_retro(param)  
    Rprod = Array{Any,1}(undef, 3)
    if length(keys(param)) == 6
    	(pi_0, pi_1) = (0.0, 0.0)
        if param["type_1"] == "a"           
            pi_0 = 1.0
        elseif param["type_1"] == "i"
            pi_1 = 1.0
        elseif param["type_1"] == "r"
            pi_0 = 1.0
        end
        local parent_1, self = param["parent_1"], param["self"]
        if parent_1 != 4        
            Rprod[parent_1] = quote
                pars[4][$self] * ((pars[$self][1])^2 * (y[$parent_1] / (pars[$self][2]))^((pars[$self][1])-1)) /
                (1 + (y[$parent_1] /pars[$self][2])^(pars[$self][1]))^2 
            end
        end
        Rprod[setdiff([1, 2, 3], [parent_1])] .= 0.0
        dXdt = quote
            (pars[$self][3]) * (0.999 * ($pi_0 + $pi_1 * (y[$parent_1]/(pars[$self][2]))^(pars[$self][1])) /
            (1+(y[$parent_1]/(pars[$self][2]))^(pars[$self][1])) + 0.001 - y[$self])
        end
    elseif length(keys(param)) == 10
    	(pi_0, pi_1, pi_2, pi_3) = zeros(4)
        if param["type_1"] == "r" && param["type_2"] == "r"     
            pi_0 = 1.0
        elseif param["type_1"] == "a" && (param["type_2"] == "a" || param["type_2"] == "i")
			pi_3 = 1.0
        elseif param["type_1"] == "a" && param["type_2"] == "r"
			pi_1 = 1.0
        elseif param["type_1"] == "r" && param["type_2"] == "i"
			pi_2 = 1.0 
        end 
        local parent_1, parent_2, self = param["parent_1"], param["parent_2"], param["self"]
        if parent_1 != 4
            Rprod[parent_1] = quote
                pars[4][$self] * (pars[$self][1])^2 * (y[$parent_1] /(pars[$self][2]))^((pars[$self][1])-1) / 
                (1 + (y[$parent_1]/pars[$self][2])^(pars[$self][1]))^2
            end
        end
        if parent_2 != 4
            Rprod[parent_2] = quote
                pars[4][$self] * (pars[$self][3])^2 * (y[$parent_2] /(pars[$self][4]))^((pars[$self][3])-1) / 
                (1 + (y[$parent_2]/pars[$self][4])^(pars[$self][3]))^2
            end
        end
        Rprod[setdiff([1, 2, 3], [parent_1, parent_2])] .= 0.0
        dXdt = quote
            (pars[$self][5]) * (0.999 * ($pi_0 + $pi_1 * (y[$parent_1]/(pars[$self][2]))^(pars[$self][1]) +
            $pi_2 * (y[$parent_2]/(pars[$self][4]))^(pars[$self][3]) + $pi_3 * (y[$parent_1]/(pars[$self][2]))^(pars[$self][1]) * 
            (y[$parent_2]/(pars[$self][4]))^(pars[$self][3])) /
            (1 + (y[$parent_1]/(pars[$self][2]))^(pars[$self][1]) + (y[$parent_2]/(pars[$self][4]))^(pars[$self][3]) +
            (y[$parent_1]/(pars[$self][2]))^(pars[$self][1]) * (y[$parent_2]/(pars[$self][4]))^(pars[$self][3])) + 0.001 - y[$self])
        end
    elseif length(keys(param)) == 14
    	(pi_0, pi_1, pi_2, pi_3, pi_12, pi_23, pi_13, pi_123) = zeros(8);
        if param["type_1"] == "r" && param["type_2"] == "r" && param["type_3"] == "r"     
        	pi_0 = 1.0;
        elseif param["type_1"] == "a" && param["type_2"] == "a" && (param["type_3"] == "a" || param["type_3"] == "i")
            pi_123 = 1.0;
        elseif param["type_1"] == "a" && param["type_2"] == "a" && param["type_3"] == "r"
            pi_12 = 1.0;
        elseif param["type_1"] == "a" && param["type_2"] == "r" && param["type_3"] == "r"
            pi_1 = 1.0;
        elseif param["type_1"] == "r" && param["type_2"] == "r" && param["type_3"] == "i"
            pi_3 = 1.0;
        elseif param["type_1"] == "a" && param["type_2"] == "r" && param["type_3"] == "i"
            pi_13 = 1.0;
        end
        local parent_1, parent_2, parent_3, self = param["parent_1"], param["parent_2"], param["parent_3"], param["self"]
        if parent_1 != 4
            Rprod[parent_1] = quote
                pars[4][$self] * (pars[$self][1])^2 * (y[$parent_1] /(pars[$self][2]))^((pars[$self][1])-1) / 
                (1 + (y[$parent_1]/pars[$self][2])^(pars[$self][1]))^2
            end
        end
        if parent_2 != 4
            Rprod[parent_2] = quote
                pars[4][$self] * (pars[$self][3])^2 * (y[$parent_2] /(pars[$self][4]))^((pars[$self][3])-1) / 
                (1 + (y[$parent_2]/pars[$self][4])^(pars[$self][3]))^2
            end
        end
        if parent_3 != 4
            Rprod[parent_3] = quote
                pars[4][$self] * (pars[$self][5])^2 * (y[$parent_3] /(pars[$self][6]))^((pars[$self][5])-1) / 
                (1 + (y[$parent_3]/pars[$self][6])^(pars[$self][5]))^2
            end        
        end
        Rprod[setdiff([1, 2, 3], [parent_1, parent_2, parent_3])] .= 0.0
        dXdt = quote
            (pars[$self][7]) * (0.999 * ($pi_0 + $pi_1 * (y[$parent_1]/(pars[$self][2]))^(pars[$self][1]) + 
            $pi_2 * (y[$parent_2]/(pars[$self][4]))^(pars[$self][3]) +
            $pi_3 * (y[$parent_3]/(pars[$self][6]))^(pars[$self][5]) + 
            $pi_12 * (y[$parent_1]/(pars[$self][2]))^(pars[$self][1]) * (y[$parent_2]/(pars[$self][4]))^(pars[$self][3]) +
            $pi_13 * (y[$parent_1]/(pars[$self][2]))^(pars[$self][1]) * (y[$parent_3]/(pars[$self][6]))^(pars[$self][5]) +
            $pi_23 * (y[$parent_2]/(pars[$self][4]))^(pars[$self][3]) * (y[$parent_3]/(pars[$self][6]))^(pars[$self][5]) +
            $pi_123 * (y[$parent_1]/(pars[$self][2]))^(pars[$self][1]) * (y[$parent_2]/(pars[$self][4]))^(pars[$self][3]) 
            * (y[$parent_3]/(pars[$self][6]))^(pars[$self][5]) ) /
            (1 + (y[$parent_1]/(pars[$self][2]))^(pars[$self][1]) + (y[$parent_2]/(pars[$self][4]))^(pars[$self][3]) + 
            (y[$parent_3]/(pars[$self][6]))^(pars[$self][5]) +
            (y[$parent_1]/(pars[$self][2]))^(pars[$self][1]) * (y[$parent_2]/(pars[$self][4]))^(pars[$self][3]) +
            (y[$parent_1]/(pars[$self][2]))^(pars[$self][1]) * (y[$parent_3]/(pars[$self][6]))^(pars[$self][5]) +
            (y[$parent_2]/(pars[$self][4]))^(pars[$self][3]) * (y[$parent_3]/(pars[$self][6]))^(pars[$self][5]) +
            (y[$parent_1]/(pars[$self][2]))^(pars[$self][1]) * (y[$parent_2]/(pars[$self][4]))^(pars[$self][3]) * 
            (y[$parent_3]/(pars[$self][6]))^(pars[$self][5]) ) + 0.001 - y[$self]) 
        end
    elseif length(keys(param)) == 18
    	(pi_0, pi_1, pi_2, pi_3, pi_4, pi_12, pi_13, pi_14, pi_23, pi_24, pi_34, pi_123,
    	 pi_124, pi_134, pi_234, pi_1234) = zeros(16)
        if param["type_1"] == "r" && param["type_2"] == "r" && param["type_3"] == "r"          
			pi_4 = 1.0
        elseif param["type_1"] == "a" && param["type_2"] == "a" && param["type_3"] == "a"
            pi_1234 = 1.0
        elseif param["type_1"] == "a" && param["type_2"] == "a" && param["type_3"] == "r"
            pi_124 = 1.0
        elseif param["type_1"] == "a" && param["type_2"] == "r" && param["type_3"] == "r" 
            pi_14 = 1.0
        end
        local parent_1, parent_2, parent_3, parent_4, self = param["parent_1"], param["parent_2"], param["parent_3"], param["parent_4"], param["self"]
        if parent_1  != 4
            Rprod[parent_1] = quote
                pars[4][$self] * (pars[$self][1])^2 * (y[$parent_1] /(pars[$self][2]))^((pars[$self][1])-1) / 
                (1 + (y[$parent_1]/pars[$self][2])^(pars[$self][1]))^2
            end
        end
        if parent_2 !== 4
            Rprod[parent_2] = quote
                pars[4][$self] * (pars[$self][3])^2 * (y[$parent_2] /(pars[$self][4]))^((pars[$self][3])-1) / 
                (1 + (y[$parent_2]/pars[$self][4])^(pars[$self][3]))^2
            end
        end
        if parent_3 != 4
            Rprod[parent_3] = quote
                pars[4][$self] * (pars[$self][5])^2 * (y[$parent_3] /(pars[$self][6]))^((pars[$self][5])-1) / 
                (1 + (y[$parent_3]/pars[$self][6])^(pars[$self][5]))^2
            end
        end
        Rprod[setdiff([1, 2, 3], [parent_1, parent_2, parent_3])] .= 0.0
        dXdt = quote
            (pars[$self][9]) * (0.999 * ($pi_0 + $pi_1 * (y[$parent_1]/(pars[$self][2]))^(pars[$self][1]) + 
            $pi_2 * (y[$parent_2]/(pars[$self][4]))^(pars[$self][3]) +
            $pi_3 * (y[$parent_3]/(pars[$self][6]))^(pars[$self][5]) + 
            $pi_4 * (y[$parent_4]/(pars[$self][8]))^(pars[$self][7]) +
            $pi_12 * (y[$parent_1]/(pars[$self][2]))^(pars[$self][1]) * (y[$parent_2]/(pars[$self][4]))^(pars[$self][3]) +
            $pi_13 * (y[$parent_1]/(pars[$self][2]))^(pars[$self][1]) * (y[$parent_3]/(pars[$self][6]))^(pars[$self][5]) +
            $pi_14 * (y[$parent_1]/(pars[$self][2]))^(pars[$self][1]) * (y[$parent_4]/(pars[$self][8]))^(pars[$self][7]) +
            $pi_23 * (y[$parent_2]/(pars[$self][4]))^(pars[$self][3]) * (y[$parent_3]/(pars[$self][6]))^(pars[$self][5]) +
            $pi_24 * (y[$parent_2]/(pars[$self][4]))^(pars[$self][3]) * (y[$parent_4]/(pars[$self][8]))^(pars[$self][7]) +
            $pi_34 * (y[$parent_3]/(pars[$self][6]))^(pars[$self][5]) * (y[$parent_4]/(pars[$self][8]))^(pars[$self][7]) +
            $pi_123 * (y[$parent_1]/(pars[$self][2]))^(pars[$self][1]) * (y[$parent_2]/(pars[$self][4]))^(pars[$self][3]) * 
            (y[$parent_3]/(pars[$self][6]))^(pars[$self][5]) +
            $pi_124 * (y[$parent_1]/(pars[$self][2]))^(pars[$self][1]) * (y[$parent_2]/(pars[$self][4]))^(pars[$self][3]) * 
            (y[$parent_4]/(pars[$self][8]))^(pars[$self][7]) +
            $pi_134 * (y[$parent_1]/(pars[$self][2]))^(pars[$self][1]) * (y[$parent_3]/(pars[$self][6]))^(pars[$self][5]) * 
            (y[$parent_4]/(pars[$self][8]))^(pars[$self][7]) +
            $pi_234 * (y[$parent_2]/(pars[$self][4]))^(pars[$self][3]) * (y[$parent_3]/(pars[$self][6]))^(pars[$self][5]) * 
            (y[$parent_4]/(pars[$self][8]))^(pars[$self][7]) +
            $pi_1234 * (y[$parent_1]/(pars[$self][2]))^(pars[$self][1]) * (y[$parent_2]/(pars[$self][4]))^(pars[$self][3]) * 
            (y[$parent_3]/(pars[$self][6]))^(pars[$self][5]) * (y[$parent_4]/(pars[$self][8]))^(pars[$self][7])) /
            (1 + (y[$parent_1]/(pars[$self][2]))^(pars[$self][1]) + (y[$parent_2]/(pars[$self][4]))^(pars[$self][3]) +
            (y[$parent_3]/(pars[$self][6]))^(pars[$self][5]) + (y[$parent_4]/(pars[$self][8]))^(pars[$self][7]) + 
            (y[$parent_1]/(pars[$self][2]))^(pars[$self][1]) * (y[$parent_2]/(pars[$self][4]))^(pars[$self][3]) +
            (y[$parent_1]/(pars[$self][2]))^(pars[$self][1]) * (y[$parent_3]/(pars[$self][6]))^(pars[$self][5]) +
            (y[$parent_1]/(pars[$self][2]))^(pars[$self][1]) * (y[$parent_4]/(pars[$self][8]))^(pars[$self][7]) + 
            (y[$parent_2]/(pars[$self][4]))^(pars[$self][3]) * (y[$parent_3]/(pars[$self][6]))^(pars[$self][5]) +
            (y[$parent_2]/(pars[$self][4]))^(pars[$self][3]) * (y[$parent_4]/(pars[$self][8]))^(pars[$self][7]) + 
            (y[$parent_3]/(pars[$self][6]))^(pars[$self][5]) * (y[$parent_4] /(pars[$self][8]))^(pars[$self][7]) +
            (y[$parent_1]/(pars[$self][2]))^(pars[$self][1]) * (y[$parent_2]/(pars[$self][4]))^(pars[$self][3]) * 
            (y[$parent_3]/(pars[$self][6]))^(pars[$self][5]) +
            (y[$parent_1]/(pars[$self][2]))^(pars[$self][1]) * (y[$parent_2]/(pars[$self][4]))^(pars[$self][3]) * 
            (y[$parent_4]/(pars[$self][8]))^(pars[$self][7]) +
            (y[$parent_1]/(pars[$self][2]))^(pars[$self][1]) * (y[$parent_3]/(pars[$self][6]))^(pars[$self][5]) * 
            (y[$parent_4]/(pars[$self][8]))^(pars[$self][7]) +
            (y[$parent_2]/(pars[$self][4]))^(pars[$self][3]) * (y[$parent_3]/(pars[$self][6]))^(pars[$self][5]) * 
            (y[$parent_4]/(pars[$self][8]))^(pars[$self][7]) +
            (y[$parent_1]/(pars[$self][2]))^(pars[$self][1]) * (y[$parent_2]/(pars[$self][4]))^(pars[$self][3]) * 
            (y[$parent_3]/(pars[$self][6]))^(pars[$self][5]) * (y[$parent_4]/(pars[$self][8]))^(pars[$self][7])) + 0.001 - y[$self])
        end
    end
    return dXdt, Rprod
end

################# combine three ODEs into an ODE system (use expressions) ###################
function create_three_node_ODE(inter_param_matrix)
    ex1, Rprod1 = one_node_ODE_retro(inter_param_matrix[1])
    ex2, Rprod2 = one_node_ODE_retro(inter_param_matrix[2])
    ex3, Rprod3 = one_node_ODE_retro(inter_param_matrix[3])
    
    # reduction factor due to retroactivity, set to 0 if not considered.
    R_total_inverse_1 = Expr(:call, :./, 1, Expr(:call, :+, Rprod1[1], Rprod2[1], Rprod3[1], 1))
    R_total_inverse_2 = Expr(:call, :./, 1, Expr(:call, :+, Rprod1[2], Rprod2[2], Rprod3[2], 1))
    R_total_inverse_3 = Expr(:call, :./, 1, Expr(:call, :+, Rprod1[3], Rprod2[3], Rprod3[3], 1))
    
    # rate of change = rate of change w/o retroactivity * reduction factor
    quote
        function three_node_ODE_inside(dy, y, pars, t)
            y = max.(y, 0)
            dy[1] = ($R_total_inverse_1) * ($ex1)
            dy[2] = ($R_total_inverse_2) * ($ex2)
            dy[3] = ($R_total_inverse_3) * ($ex3)
            dy[4] = 0.0
        end
    end
end

################  simulate the model for each topology  ########################
function master_simulate_trace_and(three_node_ODE, pars) 
    # calculate circuit behavior with low input
    y0_1 = [0.001, 0.001, 0.001, 0.00]  # initial values
    tspan = 10000.0
    prob1 = ODEProblem(three_node_ODE, y0_1, tspan, pars)
    sol1 = solve(prob1)
    
    # calculate circuit behavior with high input                   
    y0_2 = sol1.u[end]
    y0_2[end] = 10.0
    prob2 = ODEProblem(three_node_ODE, y0_2, tspan, pars)
    sol2 = solve(prob2)
    time = sol2.t
    sol2_mat = hcat(sol2.u...)
    input_trace = sol2_mat[1, :]
    intermediate_trace = sol2_mat[2, :]
    output_trace = sol2_mat[3, :]
    inducer_trace = sol2_mat[4,:]
    integration_success = true
    if repr(sol2.retcode) != ":Success"
        integration_success = false
    end     
    return (time, input_trace, intermediate_trace, output_trace, integration_success)
end


######## generating ternary representation of a number ###########
function ternary(n)
    if n == 0
        return '0'
    end
    nums = String[]
    while n>0
        r = mod(n, 3)
        n = div(n, 3)      
        push!(nums, string(r))
    end    
    return join(reverse(nums))
end


############################################################################
################################# Simulation ###############################
############################################################################

left_end = 0
right_end = 3^9-1;
#  m = range(left_end, right_end; step=1);     # ternary representations of all three-gene topologies
m = [6898, 8410, 12730, 11326];   # ternary representations of four incoherent feedforward loops: 6898 (type-II), 8410 (type-IV), 12730 (type-I), 11326 (type-III)

for i = 1:length(m)
	println("Topology #: ", m[i]);
	
	# convert the ternary number to the topology matrix    
    interaction_string = ternary(m[i]);
    interaction_array = [parse(Int64, char) for char in interaction_string];
    interaction_array = interaction_array .- 1;
    interaction_array = [-ones(9-length(interaction_array), 1); interaction_array];
    interaction_matrix = reshape(interaction_array, (3, 3));
    interaction_matrix = transpose(interaction_matrix);
    println("Topology matrix: ", interaction_matrix);
    
    # check if the topology is connected
    # if topology not connected, then proceed
    if sum(interaction_matrix[1, :].!=0) == 0 || broadcast(abs, interaction_matrix[1, :]) == [1, 0, 0] || sum(interaction_matrix[:, 2].!=0) == 0 ||
    (broadcast(abs, interaction_matrix[2, :]) == [0, 1, 0] && broadcast(abs, interaction_matrix[:, 2]) == [0, 1, 0]) ||
    sum(interaction_matrix[:, 3].!=0) == 0 || (broadcast(abs, interaction_matrix[3, :]) == [0, 0, 1] && broadcast(abs, interaction_matrix[:, 3]) == [0, 0, 1])
        continue;
    end
    
    # specify the number of parameter sets to sample
    n_pts = 10000;
    
    # eta is normalized DNA concentration; eta can be adjusted to change retroactivity
    eta_lst = [(0.0, 0.0, 0.0), (0.1, 0.1, 0.1), (1.0, 1.0, 1.0), (10.0, 10.0, 10.0)];
    
    # generate the ODE system
    inter_param_matrix = generate_rand_params(interaction_matrix);
    three_node_ODE = eval(create_three_node_ODE(inter_param_matrix));
    
    let ODE_func = three_node_ODE
		for n = 1:n_pts
			println("Trajectory #: ",  n);
			
			# generate random parameters
			inter_param_matrix = generate_rand_params(interaction_matrix);

			# convert matrix to tuples
			rand_params_tuple_lst = dict_to_tuples(inter_param_matrix);

			# simulate the ODE system at four different levels of eta
			for (j, eta_tuple) in enumerate(eta_lst)
				rand_params_tuple_lst_2 = (rand_params_tuple_lst[1], rand_params_tuple_lst[2],
				rand_params_tuple_lst[3], eta_tuple);   
				
				# plug in the ODE function and the parameters
				(time, input_trace, intermediate_trace, output_trace, integration_success) = 
				master_simulate_trace_and(ODE_func, rand_params_tuple_lst_2);
				
				# to check the behavior of the circuit, add codes heres ...
				###############
				###############
				
			end
		end
	end
end
