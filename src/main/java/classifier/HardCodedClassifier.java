/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package classifier;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author McKay
 */
public class HardCodedClassifier extends Classifier{

    @Override
    public void buildClassifier(Instances i) throws Exception {
        
    }

    @Override
    public double classifyInstance(Instance instnc) throws Exception {
        return 0;
    }

//    @Override
//    public double[] distributionForInstance(Instance instnc) throws Exception {
//        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
//    }
//
//    @Override
//    public Capabilities getCapabilities() {
//        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
//    }
    
}
