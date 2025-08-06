(set-logic SLIA)
 
(synth-fun f ((name String)) String
    ((Start String (ntString))
     (ntString String (name " "
                       (str.++ ntString ntString)
                       (str.replace ntString ntString ntString)
                       (str.at ntString ntInt)
                       (int.to.str ntInt)
                       (str.substr ntString ntInt ntInt)))
      (ntInt Int (0 1 2 3 4 5
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

(constraint (= (f "938-242-504") "242"))
(constraint (= (f "308-916-545") "916"))
(constraint (= (f "623-599-749") "599"))
(constraint (= (f "981-424-843") "424"))
(constraint (= (f "118-980-214") "980"))
(constraint (= (f "244-655-094") "655"))

(check-synth)
