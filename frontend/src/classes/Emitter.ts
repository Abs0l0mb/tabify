'use strict';

import {
    Listener,
    TListenerTriggerCallback,
    TListenerOffCallback,
    Tools
} from '@src/classes';

/**
 * Classe `Emitter`
 *
 * Fournit un système simple de **programmation par événements**.
 * 
 * `Emitter` permet à une classe de :
 * - enregistrer des écouteurs (`on`) pour réagir à des événements nommés,
 * - émettre des événements (`emit`) avec des données associées,
 * - désactiver les écouteurs automatiquement via un callback `off`.
 *
 * Elle repose sur la classe `Listener` pour représenter chaque abonnement.
 * La plupart des composants du framework (ex : `Block`, `Table`, `Api`, etc.)
 * héritent de `Emitter` pour notifier des actions (clics, chargements, mises à jour...).
 *
 * @example
 * ```typescript
 * const emitter = new Emitter();
 *
 * // Écouter un événement
 * emitter.on("ready", () => console.log("Prêt !"));
 *
 * // Déclencher un événement
 * emitter.emit("ready");
 * ```
 */

export class Emitter {

    /**
     * Liste des écouteurs enregistrés, indexés par leur identifiant unique.
     * Chaque entrée correspond à une instance de `Listener`.
     */
    private listeners: {[id: string]: Listener} = {};
    
    /**
     * Crée une nouvelle instance d’émetteur d’événements.
     */
    constructor() {
    }

    /**
     * Enregistre un nouvel écouteur sur un événement donné.
     *
     * @param {string} event - Nom de l’événement à écouter.
     * @param {TListenerTriggerCallback} triggerCallback - Fonction exécutée lorsque l’événement est émis.
     * @returns {Listener} L’objet `Listener` associé à cet abonnement.
     *
     * @example
     * ```typescript
     * const listener = emitter.on("click", (data) => console.log("Données :", data));
     * ```
     *
     * @remarks
     * Le `Listener` retourné contient un callback `off()` permettant de se désabonner facilement :
     * ```typescript
     * const l = emitter.on("refresh", fn);
     * l.off(); // retire le listener
     * ```
     */
    public on(event: string, triggerCallback: TListenerTriggerCallback) : Listener {

        const id = Tools.uid();

        const offCallback: TListenerOffCallback = () => {
            this.off(id);
        };

        const listener = new Listener(id, event, triggerCallback, offCallback);

        this.listeners[id] = listener;

        return listener;
    }

    /**
     * Supprime un écouteur à partir de son identifiant.
     *
     * @param {string} id - Identifiant unique du listener à retirer.
     *
     * @remarks
     * Cette méthode est principalement utilisée en interne via le `off()` de `Listener`.
     */
    private off(id: string) : void {

        for (const listener of Object.values(this.listeners)) {

            if (listener && listener.getId() === id) {
                delete this.listeners[id];
                return;
            }
        }
    }

    /**
     * Émet un événement et déclenche tous les écouteurs associés.
     *
     * @param {string} event - Nom de l’événement à émettre.
     * @param {any} [data] - Données optionnelles transmises aux callbacks.
     *
     * @example
     * ```typescript
     * emitter.emit("update", { id: 42 });
     * ```
     *
     * @remarks
     * Tous les `Listener` enregistrés pour cet événement recevront les mêmes données.
     */
    public emit(event: string, data?: any) : void {

        for (const listener of Object.values(this.listeners)) {

            if (listener && listener.getEvent() === event)
                listener.trigger(data);
        } 
    }

    /**
     * Supprime tous les écouteurs enregistrés.
     *
     * @protected
     * @example
     * ```typescript
     * this.clearListeners(); // appelé lors d’un "delete()" d’un composant
     * ```
     */
    protected clearListeners() : void {

        this.listeners = {};
    }
}