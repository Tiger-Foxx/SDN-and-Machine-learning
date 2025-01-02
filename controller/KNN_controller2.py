from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.lib import hub
from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp
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
        self.congestion_stats = {}
        self.qos_policies = {
            'video': {'min_rate': 10000000, 'priority': 3},  # 10 Mbps
            'voip': {'min_rate': 5000000, 'priority': 4},    # 5 Mbps
            'data': {'min_rate': 1000000, 'priority': 2}     # 1 Mbps
        }
        
        start = datetime.now()
        self.flow_training()
        end = datetime.now()
        self.logger.info("Training time: %s", (end-start))

    def _identify_traffic_type(self, flow):
        """Identification du type de trafic basée sur les ports"""
        if flow['tp_dst'] in [80, 443]:  # Web traffic
            return 'data'
        elif flow['tp_dst'] in [5060, 5061]:  # VoIP
            return 'voip'
        elif flow['tp_dst'] in [1935, 554]:  # Streaming
            return 'video'
        return 'data'  # Default

    def _check_congestion(self, datapath_id, port_no):
        """Vérification de la congestion sur un port"""
        stats = self.congestion_stats.get((datapath_id, port_no), {})
        if stats:
            utilization = stats['bytes'] / (stats['timestamp'] - stats['last_timestamp'])
            return utilization > 0.8  # 80% utilisation = congestion
        return False

    def _handle_congestion(self, datapath, flow_data):
        """Gestion de la congestion"""
        self.logger.info("Handling congestion for datapath %s", datapath.id)
        
        # Identifier les flux prioritaires
        traffic_type = self._identify_traffic_type(flow_data)
        policy = self.qos_policies[traffic_type]
        
        if policy['priority'] >= 3:  # Flux prioritaire
            # Trouver un chemin alternatif
            alternative_path = self._find_alternative_path(datapath, flow_data)
            if alternative_path:
                self._install_path_flows(datapath, flow_data, alternative_path, policy)
                self.logger.info("Rerouted high priority flow via alternative path")
        else:
            # Réduire le débit des flux non prioritaires
            self._apply_rate_limiting(datapath, flow_data)
            self.logger.info("Applied rate limiting to low priority flow")

    def _apply_rate_limiting(self, datapath, flow):
        """Application de la limitation de débit"""
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto
        
        match = parser.OFPMatch(
            eth_type=0x0800,
            ipv4_src=flow['ip_src'],
            ipv4_dst=flow['ip_dst']
        )
        
        # Configuration de la limitation de débit
        band = parser.OFPMeterBandDrop(rate=1000000)  # 1 Mbps
        mod = parser.OFPMeterMod(
            datapath=datapath,
            command=ofproto.OFPMC_ADD,
            flags=ofproto.OFPMF_KBPS,
            meter_id=1,
            bands=[band]
        )
        datapath.send_msg(mod)
        
        # Associer le flux au meter
        actions = [parser.OFPActionOutput(ofproto.OFPP_NORMAL)]
        inst = [
            parser.OFPInstructionMeter(1),
            parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)
        ]
        
        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=1,
            match=match,
            instructions=inst
        )
        datapath.send_msg(mod)

    def _find_alternative_path(self, datapath, flow):
        """Recherche d'un chemin alternatif moins congestionné"""
        src_sw = self._get_switch_for_ip(flow['ip_src'])
        dst_sw = self._get_switch_for_ip(flow['ip_dst'])
        
        if not (src_sw and dst_sw):
            return None
            
        # Utiliser Dijkstra avec poids basés sur la congestion
        def weight_func(u, v, d):
            port = d.get('port')
            if self._check_congestion(u, port):
                return 1000  # Pénaliser les liens congestionnés
            return 1
            
        try:
            path = nx.shortest_path(self.topology_graph, src_sw, dst_sw, 
                                  weight=weight_func)
            if len(path) > 1:  # Vérifier que le chemin est valide
                return path
        except:
            pass
        return None

    def _install_path_flows(self, datapath, flow, path, policy):
        """Installation des règles QoS sur le chemin"""
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto
        
        # Configurer le meter pour la QoS
        band = parser.OFPMeterBandDrop(rate=policy['min_rate'])
        mod = parser.OFPMeterMod(
            datapath=datapath,
            command=ofproto.OFPMC_ADD,
            flags=ofproto.OFPMF_KBPS,
            meter_id=policy['priority'],
            bands=[band]
        )
        datapath.send_msg(mod)
        
        # Installer les règles de flux pour chaque switch du chemin
        for i in range(len(path) - 1):
            dp = self.datapaths[path[i]]
            out_port = self.topology_graph[path[i]][path[i+1]]['port']
            
            match = parser.OFPMatch(
                eth_type=0x0800,
                ipv4_src=flow['ip_src'],
                ipv4_dst=flow['ip_dst']
            )
            
            actions = [parser.OFPActionOutput(out_port)]
            inst = [
                parser.OFPInstructionMeter(policy['priority']),
                parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)
            ]
            
            mod = parser.OFPFlowMod(
                datapath=dp,
                priority=policy['priority'],
                match=match,
                instructions=inst
            )
            dp.send_msg(mod)
            self.logger.info("Installed QoS flow on switch %s: %s -> port %s", 
                           dp.id, flow['ip_src'], out_port)

    def flow_predict(self):
        """Version améliorée avec QoS et gestion de congestion"""
        self.logger.info("=== Starting Enhanced Flow Prediction ===")
        try:
            predict_flow_dataset = pd.read_csv('PredictFlowStatsfile.csv')
            
            # Prédiction DDoS
            y_flow_pred = self.flow_model.predict(X_predict_flow)
            
            # Stats de trafic
            legitimate_traffic = sum(1 for y in y_flow_pred if y == 0)
            ddos_traffic = len(y_flow_pred) - legitimate_traffic
            
            if (legitimate_traffic/len(y_flow_pred)*100) > 80:
                self.logger.info("Normal Traffic Pattern - Optimizing flows")
                for _, flow in predict_flow_dataset.iterrows():
                    dp = self.datapaths.get(flow['datapath_id'])
                    if dp:
                        # Vérifier la congestion
                        if self._check_congestion(dp.id, flow['out_port']):
                            self._handle_congestion(dp, flow)
                        else:
                            # Appliquer QoS normal
                            traffic_type = self._identify_traffic_type(flow)
                            policy = self.qos_policies[traffic_type]
                            self._install_qos_flow(dp, flow, policy)
            else:
                self.logger.info("DDoS Attack - Implementing protective measures")
                self._mitigate_ddos_attack(predict_flow_dataset)
                
        except Exception as e:
            self.logger.error("Enhanced prediction error: %s", e)
            
    def _monitor(self):
        """Monitoring amélioré avec statistiques de port"""
        while True:
            self.logger.info("=== Starting monitoring cycle ===")
            for dp in self.datapaths.values():
                self._request_stats(dp)
                self._request_port_stats(dp)
            hub.sleep(10)
            self.flow_predict()

    def _request_port_stats(self, datapath):
        """Demande des statistiques de port"""
        parser = datapath.ofproto_parser
        req = parser.OFPPortStatsRequest(datapath, 0, datapath.ofproto.OFPP_ANY)
        datapath.send_msg(req)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _port_stats_reply_handler(self, ev):
        """Traitement des statistiques de port"""
        body = ev.msg.body
        
        for stat in body:
            key = (ev.msg.datapath.id, stat.port_no)
            self.congestion_stats[key] = {
                'bytes': stat.tx_bytes + stat.rx_bytes,
                'timestamp': datetime.now().timestamp(),
                'last_timestamp': self.congestion_stats.get(key, {}).get('timestamp', 0)
            }