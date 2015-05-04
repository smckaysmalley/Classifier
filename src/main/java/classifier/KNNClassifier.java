/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier;

import static java.lang.Math.*;
import java.util.*;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author McKay
 */
public class KNNClassifier extends Classifier{

    
    private Integer k;
    private Instances data;
    private TreeMap<Double,Integer> map;

    public KNNClassifier() {
        this.k = 5;
    }
    
    public KNNClassifier(int numK) {
        this.k = numK;
    }
    
    @Override
    public void buildClassifier(Instances instances) throws Exception {
        data = instances;
    }
    
    @Override
    public double classifyInstance(Instance instance) throws Exception {
    
        createMap(instance);
        
        return classify(instance);
    }
    /**
     * 
     * @param instance
     * @return 
     */
    double classify(Instance instance){
        int tally[] = new int[data.numClasses()];
        
        int i = 0;
        for(Double key : map.keySet()){
            tally[map.get(key)]++;
            if (i >= k)
                break;
            i++;
        }
            
        int highestValue = 0;
        int highestIndex = 0;
        i = 0;
        for(int value : tally){          

            if(highestValue < value){
                highestValue = value;
                highestIndex = i;
            }
            i++;
        }

        return highestIndex;
    }
    /**
     * 
     * @param first
     * @param second
     * @return 
     */
    double findDistance(Instance first, Instance second)
    {
        double distance = 0;
        for (int i = 0; i < first.numAttributes()-1 && i < second.numAttributes()-1; ++i)
        {
            if (first.attribute(i).isNumeric() && second.attribute(i).isNumeric())
            {
                distance += pow(first.value(i) - second.value(i), 2);
            }
            else
            {
                if (first.stringValue(i).equals(second.stringValue(i)))
                {
                    distance += 0;
                }
                distance += k;
            }
        }
        
        return distance;
    }
    
    /**
     * 
     * @param instance 
     */
    void createMap(Instance instance){
        map = new TreeMap<>();
        
        for(int i = 0; i < data.numInstances(); i++)
            map.put(findDistance(data.instance(i), instance), (int)(data.instance(i).classValue()));
    }
    
}


