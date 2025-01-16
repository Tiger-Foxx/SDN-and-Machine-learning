from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub
import os
import switch
from datetime import datetime


import time




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import joblib
from sklearn.metrics import accuracy_score








class SimpleMonitor13(switch.SimpleSwitch13):








  def __init__(self, *args, **kwargs):
       super(SimpleMonitor13, self).__init__(*args, **kwargs)
       self.datapaths = {}
       self.monitor_thread = hub.spawn(self._monitor)
       self.blocking_thread = None
       self.attackers = set()  # List of attackers detected




       start = datetime.now()
       self.flow_training()
       end = datetime.now()
       print("Training time: ", (end-start))








  @set_ev_cls(ofp_event.EventOFPStateChange,
              [MAIN_DISPATCHER, DEAD_DISPATCHER])
  def _state_change_handler(self, ev):
      datapath = ev.datapath
      if ev.state == MAIN_DISPATCHER:
          if datapath.id not in self.datapaths:
              self.logger.debug('register datapath: %016x', datapath.id)
              self.datapaths[datapath.id] = datapath
      elif ev.state == DEAD_DISPATCHER:
          if datapath.id in self.datapaths:
              self.logger.debug('unregister datapath: %016x', datapath.id)
              del self.datapaths[datapath.id]








  def _monitor(self):
      while True:
          for dp in self.datapaths.values():
              self._request_stats(dp)
          hub.sleep(10)








          self.flow_predict()








  def _request_stats(self, datapath):
      self.logger.debug('send stats request: %016x', datapath.id)
      parser = datapath.ofproto_parser








      req = parser.OFPFlowStatsRequest(datapath)
      datapath.send_msg(req)








  @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
  def _flow_stats_reply_handler(self, ev):








      timestamp = datetime.now()
      timestamp = timestamp.timestamp()








      file0 = open("PredictFlowStatsfile.csv","w")
      file0.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')
      body = ev.msg.body
      icmp_code = -1
      icmp_type = -1
      tp_src = 0
      tp_dst = 0








      # Filtrer les flux avec sécurité
      filtered_flows = []
      for flow in body:
       if flow.priority <= 10:
           # Élargir la plage de priorités
           # Vérifier si le flux a tous les champs nécessaires
           if all(key in flow.match for key in ['eth_type', 'ipv4_src', 'ipv4_dst', 'ip_proto']):
               filtered_flows.append(flow)
       # Trier les flux filtrés
      for stat in sorted(filtered_flows, key=lambda flow: ( flow.match['eth_type'], flow.match['ipv4_src'], flow.match['ipv4_dst'], flow.match['ip_proto'] )):
   
          ip_src = stat.match['ipv4_src']
          ip_dst = stat.match['ipv4_dst']
          ip_proto = stat.match['ip_proto']
       
          if stat.match['ip_proto'] == 1:
              icmp_code = stat.match['icmpv4_code']
              icmp_type = stat.match['icmpv4_type']
           
          elif stat.match['ip_proto'] == 6:
              tp_src = stat.match['tcp_src']
              tp_dst = stat.match['tcp_dst']








          elif stat.match['ip_proto'] == 17:
              tp_src = stat.match['udp_src']
              tp_dst = stat.match['udp_dst']








          flow_id = str(ip_src) + str(tp_src) + str(ip_dst) + str(tp_dst) + str(ip_proto)
     
          try:
              packet_count_per_second = stat.packet_count/stat.duration_sec
              packet_count_per_nsecond = stat.packet_count/stat.duration_nsec
          except:
              packet_count_per_second = 0
              packet_count_per_nsecond = 0
           
          try:
              byte_count_per_second = stat.byte_count/stat.duration_sec
              byte_count_per_nsecond = stat.byte_count/stat.duration_nsec
          except:
              byte_count_per_second = 0
              byte_count_per_nsecond = 0
           
          file0.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n"
              .format(timestamp, ev.msg.datapath.id, flow_id, ip_src, tp_src,ip_dst, tp_dst,
                      stat.match['ip_proto'],icmp_code,icmp_type,
                      stat.duration_sec, stat.duration_nsec,
                      stat.idle_timeout, stat.hard_timeout,
                      stat.flags, stat.packet_count,stat.byte_count,
                      packet_count_per_second,packet_count_per_nsecond,
                      byte_count_per_second,byte_count_per_nsecond))
       
      file0.close()
















  def flow_training(self):
      self.logger.info("Flow Training (loading pre-trained model) ...")








      try:
          # Charger le modèle enregistré
          self.flow_model = joblib.load('flow_model.pkl')
          self.logger.info("Model successfully loaded from 'flow_model.pkl'")
      except FileNotFoundError:
          self.logger.error("Model file 'flow_model.pkl' not found. Please train the model separately.")
          return
      except Exception as e:
          self.logger.error(f"An error occurred while loading the model: {e}")
          return








      self.logger.info("Model is now loaded and ready to use.")
      self.logger.info("------------------------------------------------------------------------------")




  def flow_predict(self):
   try:
       # Charger les données de prédiction
       predict_flow_dataset = pd.read_csv('PredictFlowStatsfile.csv')
       predict_flow_dataset_copy=predict_flow_dataset.copy()
      


      
       # Nettoyer les colonnes spécifiques pour retirer les caractères indésirables
       predict_flow_dataset.iloc[:, 2] = predict_flow_dataset.iloc[:, 2].str.replace('.', '')
       predict_flow_dataset.iloc[:, 3] = predict_flow_dataset.iloc[:, 3].str.replace('.', '')
       predict_flow_dataset.iloc[:, 5] = predict_flow_dataset.iloc[:, 5].str.replace('.', '')




       # Préparer les données pour la prédiction
       X_predict_flow = predict_flow_dataset.iloc[:, :].values
       X_predict_flow = X_predict_flow.astype('float64')




       # Effectuer la prédiction
       y_flow_pred = self.flow_model.predict(X_predict_flow)




       # Initialiser les compteurs
       legitimate_trafic = 0
       ddos_trafic = 0
       victim = None
       self.attackers = []  # Réinitialiser la liste des attaquants




       # Analyser les résultats des prédictions
       for idx, prediction in enumerate(y_flow_pred):
           if prediction == 0:
               legitimate_trafic += 1
           else:
               ddos_trafic += 1
               original_attaker = predict_flow_dataset_copy.iloc[idx, 3]
               original_victim= predict_flow_dataset_copy.iloc[idx, 5]


               # Identifier l'attaquant et la victime
               attacker_ip = original_attaker  # IP source de l'attaque
               victim_ip = original_victim  # IP destination de l'attaque
               self.attackers.append(attacker_ip)




               # Calculer l'hôte victime
               victim = original_victim  # Remplace cette logique si elle ne s'applique pas




       # Afficher les statistiques de trafic
       self.logger.info("------------------------------------------------------------------------------")
       traffic_legitimacy = (legitimate_trafic / len(y_flow_pred)) * 100




       if traffic_legitimacy > 80:
           self.logger.info(f"Legitimate traffic detected: {traffic_legitimacy:.2f}% source : {original_attaker}")
       else:
           self.logger.info(f"DDoS traffic detected: {100 - traffic_legitimacy:.2f}%")
           if victim is not None:
               self.logger.info(f"Victim is host: h{victim}  source ---{original_attaker}---")
           if self.attackers:
               self.logger.info(f"Attackers identified: {', '.join(self.attackers)}")




       self.logger.info("------------------------------------------------------------------------------")




       # Appliquer les règles de blocage pour les attaquants
       if self.attackers:
           self.logger.info("Initiating blocking rules for detected attackers...")
           self._apply_blocking_rules()




       # Réinitialiser le fichier de prédiction
       with open("PredictFlowStatsfile.csv", "w") as file0:
           file0.write(
               'timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n'
           )
   except Exception as e:
       self.logger.error(f"No flow prediction Empty data for this flow flow prediction: {e}")




  


  def _apply_blocking_rules(self):
       self.logger.info("Starting periodic blocking rules for attackers...")


       def block_attackers():
           """Cette fonction gère le blocage des attaquants en tâche de fond."""
           start_time = time.time()  # Temps de départ pour contrôler la durée


           while True:
               # Vérifiez si 5 minutes se sont écoulées
               if time.time() - start_time > 300:  # 300 secondes = 5 minutes
                   self.logger.info("5 minutes are over, stopping blocking rules.")
                   break  # Sortir de la boucle après 5 minutes


               if self.attackers:
                   for attacker_ip in self.attackers:
                       for datapath in self.datapaths.values():
                           parser = datapath.ofproto_parser
                           ofproto = datapath.ofproto


                           match = parser.OFPMatch(ipv4_src=attacker_ip, eth_type=0x0800)
                           actions = []  # Drop packet
                           inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
                           mod = parser.OFPFlowMod(datapath=datapath, priority=100, match=match, instructions=inst, idle_timeout=3, hard_timeout=3)
                           datapath.send_msg(mod)
                           self.logger.info(f"Blocking traffic from {attacker_ip}")


               hub.sleep(5)  # Attendre 5 secondes avant de recommencer le cycle


       # Lancer cette tâche de blocage en parallèle
       hub.spawn(block_attackers)
       self.logger.info("Blocking rules are now being applied periodically.")
























