'use strict';

export type TListenerTriggerCallback = (data: any) => void;
export type TListenerOffCallback = () => void;


/**
 * Classe `Listener`
 *
 * Représente un **écouteur d’événement** créé par la classe `Emitter`.
 * 
 * Chaque instance de `Listener` est associée à :
 * - un identifiant unique (`id`),
 * - un nom d’événement (`event`),
 * - une fonction de rappel (`triggerCallback`) à exécuter lorsque l’événement est émis,
 * - et une fonction de désinscription (`offCallback`) permettant de retirer ce listener.
 * 
 * Les objets `Listener` ne sont généralement pas créés directement par l’utilisateur.
 * Ils sont retournés par la méthode `Emitter.on()` et permettent de contrôler
 * l’abonnement à un événement.
 * 
 * @example
 * ```typescript
 * const emitter = new Emitter();
 * const listener = emitter.on("update", (data) => console.log("Mise à jour :", data));
 *
 * // Désactivation manuelle de l’écouteur
 * listener.off();
 * ```
 */

export class Listener {

    /**
     * Crée un nouveau listener associé à un événement.
     *
     * @param {string} id - Identifiant unique du listener.
     * @param {string} event - Nom de l’événement à écouter.
     * @param {TListenerTriggerCallback} triggerCallback - Fonction appelée lorsque l’événement est émis.
     * @param {TListenerOffCallback} offCallback - Fonction appelée pour désactiver ce listener.
     */ 
    constructor(
        private id: string,
        private event: string,
        private triggerCallback: TListenerTriggerCallback, 
        private offCallback: TListenerOffCallback
    ) {}

    /**
     * Retourne l’identifiant unique du listener.
     *
     * @returns {string} L’identifiant du listener.
     */
    public getId() {

        return this.id;
    }

    /**
     * Retourne le nom de l’événement auquel ce listener est abonné.
     *
     * @returns {string} Nom de l’événement écouté.
     */
    public getEvent() {

        return this.event;
    }


    /**
     * Déclenche la fonction de rappel associée à l’événement.
     *
     * @param {any} data - Données transmises lors de l’émission de l’événement.
     *
     * @example
     * ```typescript
     * listener.trigger({ id: 42 }); // exécute la callback passée à Emitter.on()
     * ```
     */
    public trigger(data: any) {

        this.triggerCallback(data);
    }

    /**
     * Désactive ce listener en appelant sa fonction de désinscription.
     *
     * @example
     * ```typescript
     * const listener = emitter.on("click", fn);
     * listener.off(); // supprime l’écoute de l’événement
     * ```
     */
    public off() : void {

        this.offCallback();
    }
}