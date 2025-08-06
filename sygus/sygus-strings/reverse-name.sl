(set-logic SLIA)
 
(synth-fun f ((firstname String) (lastname String)) String
    ((Start String (ntString))
     (ntString String (firstname lastname " "
                       (str.++ ntString ntString)
                       (str.replace ntString ntString ntString)
                       (str.at ntString ntInt)
                       (int.to.str ntInt)
                       (str.substr ntString ntInt ntInt)))
      (ntInt Int (0 1 2
                  (+ ntInt ntInt)
                  (- ntInt ntInt)
                  (str.len ntString)
                  (str.to.int ntString)
                  (str.indexof ntString ntString ntInt)))
      (ntBool Bool (true false
                    (str.prefixof ntString ntString)
                    (str.suffixof ntString ntString)
                    (str.contains ntString ntString)))))


(declare-var firstname String)
(declare-var lastname String)

(constraint (= (f "Launa" "Withers") "Withers Launa"))
(constraint (= (f "Lakenya" "Edison") "Edison Lakenya"))
(constraint (= (f "Brendan" "Hage") "Hage Brendan"))
(constraint (= (f "Bradford" "Lango") "Lango Bradford"))
(constraint (= (f "Rudolf" "Akiyama") "Akiyama Rudolf"))
(constraint (= (f "Lara" "Constable") "Constable Lara"))
 
(check-synth)
