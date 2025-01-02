from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub
import os
import switch
from datetime import datetime
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib
import networkx as nx

class SimpleMonitor13(switch.SimpleSwitch13):
    def __init__(self, *args, **kwargs):
        super(SimpleMonitor13, self).__init__(*args, **kwargs)
        self.datapaths = {}
        self.monitor_thread = hub.spawn(self._monitor)
        self.topology_graph = nx.DiGraph()
        
        start = datetime.now()
        self.flow_training()
        end = datetime.now()
        self.logger.info("Training time: %s", (end-start))
        
        # Configuration
        self.DDOS_THRESHOLD = 80  # % de trafic malveillant
        self.PACKET_RATE_THRESHOLD = 1000  # paquets/sec
        
    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def _state_change_handler(self, ev):
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.info('Register datapath: %016x', datapath.id)
                self.datapaths[datapath.id] = datapath
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.info('Unregister datapath: %016x', datapath.id)
                del self.datapaths[datapath.id]

    def _monitor(self):
        while True:
            self.logger.info("Starting monitoring cycle...")
            for dp in self.datapaths.values():
                self._request_stats(dp)
            hub.sleep(10)
            self.flow_predict()

    def _request_stats(self, datapath):
        self.logger.info('Sending stats request: %016x', datapath.id)
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _flow_stats_reply_handler(self, ev):
        timestamp = datetime.now()
        self.logger.info("Received flow stats at %s", timestamp)
        
        file0 = open("PredictFlowStatsfile.csv","w")
        file0.write('timestamp,datapath_id,flow_id,ip_src,tp_src,ip_dst,tp_dst,ip_proto,icmp_code,icmp_type,flow_duration_sec,flow_duration_nsec,idle_timeout,hard_timeout,flags,packet_count,byte_count,packet_count_per_second,packet_count_per_nsecond,byte_count_per_second,byte_count_per_nsecond\n')
        
        body = ev.msg.body
        icmp_code = -1
        icmp_type = -1
        tp_src = 0
        tp_dst = 0

        self.logger.info('datapath         ip_src            ip_dst            packets  bytes')
        self.logger.info('---------------- ---------------- ---------------- -------- --------')
        
        for stat in sorted([flow for flow in body if (flow.priority == 1)], 
                         key=lambda flow: (flow.match['eth_type'],
                                         flow.match['ipv4_src'],
                                         flow.match['ipv4_dst'],
                                         flow.match['ip_proto'])):
            
            ip_src = stat.match['ipv4_src']
            ip_dst = stat.match['ipv4_dst']
            ip_proto = stat.match['ip_proto']
            
            self.logger.info('%016x %16s %16s %8d %8d',
                           ev.msg.datapath.id, ip_src, ip_dst,
                           stat.packet_count, stat.byte_count)
            
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
                .format(timestamp, ev.msg.datapath.id, flow_id, ip_src, tp_src,
                        ip_dst, tp_dst, stat.match['ip_proto'],icmp_code,icmp_type,
                        stat.duration_sec, stat.duration_nsec,
                        stat.idle_timeout, stat.hard_timeout,
                        stat.flags, stat.packet_count,stat.byte_count,
                        packet_count_per_second,packet_count_per_nsecond,
                        byte_count_per_second,byte_count_per_nsecond))
        
        file0.close()

    def flow_training(self):
        self.logger.info("=== Starting Flow Training ===")
        try:
            self.flow_model = joblib.load('flow_model.pkl')
            self.logger.info("Model successfully loaded from 'flow_model.pkl'")
        except Exception as e:
            self.logger.error("Error loading model: %s", e)
            return
        self.logger.info("=== Flow Training Complete ===")

    def flow_predict(self):
        self.logger.info("=== Starting Flow Prediction ===")
        try:
            predict_flow_dataset = pd.read_csv('PredictFlowStatsfile.csv')
            self.logger.info("Loaded prediction dataset with %d flows", len(predict_flow_dataset))

            # Nettoyage des données
            predict_flow_dataset.iloc[:, 2] = predict_flow_dataset.iloc[:, 2].str.replace('.', '')
            predict_flow_dataset.iloc[:, 3] = predict_flow_dataset.iloc[:, 3].str.replace('.', '')
            predict_flow_dataset.iloc[:, 5] = predict_flow_dataset.iloc[:, 5].str.replace('.', '')

            X_predict_flow = predict_flow_dataset.iloc[:, :].values
            X_predict_flow = X_predict_flow.astype('float64')
            
            y_flow_pred = self.flow_model.predict(X_predict_flow)
            
            legitimate_traffic = 0
            ddos_traffic = 0
            victims = set()

            for i, prediction in enumerate(y_flow_pred):
                if prediction == 0:
                    legitimate_traffic += 1
                else:
                    ddos_traffic += 1
                    victim = int(predict_flow_dataset.iloc[i, 5]) % 20
                    victims.add(victim)
                    self._handle_ddos_flow(predict_flow_dataset.iloc[i])

            self.logger.info("=== Traffic Analysis ===")
            self.logger.info("Legitimate Traffic: %d flows (%d%%)", 
                           legitimate_traffic, 
                           (legitimate_traffic/len(y_flow_pred)*100))
            self.logger.info("DDoS Traffic: %d flows (%d%%)", 
                           ddos_traffic, 
                           (ddos_traffic/len(y_flow_pred)*100))

            if (legitimate_traffic/len(y_flow_pred)*100) > 80:
                self.logger.info("STATUS: Normal Traffic Pattern")
                self._optimize_legitimate_flows(predict_flow_dataset)
            else:
                self.logger.info("STATUS: DDoS Attack Detected!")
                self.logger.info("Targeted hosts: %s", victims)
                self._mitigate_ddos_attack(predict_flow_dataset, victims)

        except Exception as e:
            self.logger.error("Prediction error: %s", e)
        finally:
            self.logger.info("=== Flow Prediction Complete ===")

    def _handle_ddos_flow(self, flow):
        """Gestion d'un flux DDoS détecté"""
        self.logger.info("Blocking malicious flow from %s to %s", 
                        flow['ip_src'], flow['ip_dst'])
        datapath = self.datapaths.get(flow['datapath_id'])
        if datapath:
            self._install_blocking_flow(datapath, flow)

    def _optimize_legitimate_flows(self, flow_dataset):
        """
        Optimisation des flux légitimes en réorganisant les chemins pour éviter les goulets d'étranglement.
        """
        self.logger.info("Starting flow optimization for legitimate traffic...")
        
        for dp in self.datapaths.values():
            self._calculate_optimal_paths(dp, flow_dataset)
        
        self.logger.info("Flow optimization complete.")

    def _mitigate_ddos_attack(self, flow_dataset, victims):
        """Mitigation complète des attaques DDoS"""
        self.logger.info("Implementing DDoS mitigation measures...")
        for victim in victims:
            self.logger.info("Protecting host h%d...", victim)
            self._setup_victim_protection(victim)
        self.logger.info("DDoS mitigation complete")

    def _install_blocking_flow(self, datapath, flow):
        """Installation des règles de blocage"""
        parser = datapath.ofproto_parser
        match = parser.OFPMatch(
            eth_type=0x0800,
            ipv4_src=flow['ip_src'],
            ipv4_dst=flow['ip_dst']
        )
        self.add_flow(datapath, 2, match, [])  # Priorité 2, pas d'actions = drop
        self.logger.info("Installed blocking rule for %s", flow['ip_src'])

    def _calculate_optimal_paths(self, datapath, flow_dataset):
        """
        Calcul des chemins optimaux en fonction de la topologie du réseau et de l'utilisation actuelle des liens.
        """
        self.logger.info("Calculating optimal paths for datapath %x", datapath.id)
        
        try:
            # Charger la topologie actuelle du réseau
            graph = self.topology_graph.copy()
            
            # Mise à jour des poids des liens en fonction du trafic actuel
            for u, v, data in graph.edges(data=True):
                data['weight'] = 1  # Poids par défaut
                # Simulation : augmenter le poids pour les liens saturés
                link_traffic = self._get_link_traffic(u, v)
                if link_traffic > self.PACKET_RATE_THRESHOLD:
                    data['weight'] += link_traffic / self.PACKET_RATE_THRESHOLD
            
            # Calculer les chemins les plus courts (pondérés)
            for flow in flow_dataset.itertuples():
                try:
                    src = flow.ip_src
                    dst = flow.ip_dst
                    path = nx.shortest_path(graph, source=src, target=dst, weight='weight')
                    self.logger.info("Optimal path for flow %s -> %s: %s", src, dst, path)
                    # Appliquer le chemin optimal
                    self._install_optimal_path(datapath, path, flow)
                except nx.NetworkXNoPath:
                    self.logger.warning("No path found for flow %s -> %s", flow.ip_src, flow.ip_dst)
        except Exception as e:
            self.logger.error("Error calculating optimal paths: %s", e)
