(set-logic SLIA)
 
(synth-fun f ((name String)) String
    ((Start String (ntString))
     (ntString String (name " " 
                       (str.++ ntString ntString)
                       (str.replace ntString ntString ntString)
                       (str.at ntString ntInt)
                       (int.to.str ntInt)
                       (str.substr ntString ntInt ntInt)))
      (ntInt Int (0 1
                  (+ ntInt ntInt)
                  (- ntInt ntInt)
                  (str.len ntString)
                  (str.to.int ntString)
                  (str.indexof ntString ntString ntInt)))
      (ntBool Bool (true false
                    (str.prefixof ntString ntString)
                    (str.suffixof ntString ntString)
                    (str.contains ntString ntString)))))


(declare-var name String)
 
(constraint (= (f "Nancy FreeHafer") "FreeHafer"))
(constraint (= (f "Andrew Cencici") "Cencici"))
(constraint (= (f "Jan Kotas") "Kotas"))
(constraint (= (f "Mariya Sergienko") "Sergienko"))
 
(check-synth)
