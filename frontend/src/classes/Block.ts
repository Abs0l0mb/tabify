'use strict';

import {
    Emitter,
    Tools
} from '@src/classes';


/**
 * Classe `Block`
 *
 * Cette classe encapsule un `HTMLElement` pour offrir une interface orientée objet
 * plus intuitive et fluide lors de la manipulation du DOM.
 *
 * Elle hérite d’un `Emitter`, ce qui lui permet d’émettre et d’écouter des événements personnalisés.
 *
 * @example
 * ```typescript
 * const div = new Block('div', { class: 'container' });
 * div.setStyle('color', 'red')
 *    .html('<p>Hello world</p>')
 *    .appendTo(document.body);
 * ```
 *
 * @remarks
 * `Block` agit comme un wrapper simplifié autour des éléments du DOM tout en ajoutant
 * des fonctionnalités pratiques (gestion des classes, attributs, styles, écouteurs d’événements, etc.).
 */
export class Block extends Emitter {

    /**
     * Élément DOM encapsulé.
     */
    public element: any /*HTMLElement*/;

    /**
     * Identifiant unique généré automatiquement pour chaque instance.
     */
    public uid: string = Tools.uid();

    /**
     * Données publiques associées au bloc, librement définissables par l’utilisateur.
     */
    public publicData: any = {};
    
    /**
     * Crée un nouveau `Block`.
     *
     * @param {string | HTMLElement | DocumentFragment} tag - Tag HTML ou élément existant à encapsuler.
     * @param {object | string} [attributes] - Attributs à appliquer à l’élément (ou classe CSS si string).
     * @param {Block | HTMLElement} [parent] - Élément parent auquel attacher ce bloc dès la création.
     *
     * @example
     * ```typescript
     * const button = new Block('button', { class: 'primary' }, document.body);
     * button.write('Click me!');
     * ```
     */
    constructor(private tag: any,
                private attributes?: any,
                parent?: any /*Block | HTMLElement*/) {

        super();

        this.element = this.tag instanceof Node ? this.tag : document.createElement(this.tag);
        
        if (typeof this.attributes === 'string')
            this.attributes = { class: attributes };
        
        if (this.attributes)
            this.setAttributes(this.attributes);
        
        if (parent)
            this.appendTo(parent);
    }

    /** @returns La valeur d’un attribut HTML donné. */
    public getAttribute(key: string) : any {

        return this.element.getAttribute(key);
    }

    /**
     * Définit un attribut HTML.
     * @returns L’instance actuelle (`this`) pour chaînage fluide.
     */
    public setAttribute(key: any, value: any) : Block {
    
        this.element.setAttribute(key, value);

        return this;
    }

    /**
     * Définit plusieurs attributs HTML à partir d’un objet clé/valeur.
     * @returns L’instance actuelle (`this`).
     */
    public setAttributes(attributes: {[key: string] : string}) : Block {
        
        for (let key in attributes)
            this.setAttribute(key, attributes[key]);

        return this;
    }

    /** Supprime un attribut HTML. */
    public removeAttribute(key: string) : Block {

        this.element.removeAttribute(key);

        return this;
    }

    /**
     * Définit un attribut `data-*`.
     * @example block.setData('id', 123);
     */
    public setData(key: string, value: string | number) : Block {

        this.setAttribute(`data-${key}`, value);

        return this;
    }

    /** Récupère la valeur d’un attribut `data-*`. */
    public getData(key: string) : any {

        return this.element.getAttribute(`data-${key}`)
    }

    /** Retourne la valeur d’une propriété CSS calculée. */
    public getStyle(key: string) : any {

        return getComputedStyle(this.element).getPropertyValue(key);
    }

    /**
     * Définit une propriété CSS en ligne.
     * @example block.setStyle('color', 'red');
     */
    public setStyle(key: string, value: string) : Block {

        this.element.style.setProperty(key, value);
        
        return this;
    }

    /** Définit plusieurs propriétés CSS d’un coup. */
    public setStyles(styles: {[key: string]: string}) : Block {

        for (let key in styles)
            this.setStyle(key, styles[key]);

        return this;
    }

    /**
     * Remplace le contenu HTML du bloc.
     * @example block.html('<p>Hello</p>');
     */
    public html(html: string) : Block {

        this.empty();
        this.element.innerHTML = html;

        return this;
    }

    /** Supprime tous les enfants de l’élément. */
    public empty() : Block {

        while(this.element.firstChild)
            this.element.removeChild(this.element.firstChild);

        return this;
    }

    /** Vérifie si l’élément est vide (`innerHTML === ''`). */
    public isEmpty() : boolean {

        return this.element.innerHTML === '';
    }

    /** Écrit du texte brut à l’intérieur de l’élément. */
    public write(input: any) : Block {

        this.element.innerText = input;

        return this;  
    }

    /**
     * Ajoute un enfant à la fin de l’élément.
     * Émet l’événement personnalisé `append`.
     */
    public append(child: any /*Block | HTMLElement*/) : Block { 

        if (child instanceof Block)
            this.element.append(child.element);
        else if (child instanceof HTMLElement)
            this.element.append(child);

        this.emit('append');

        return this;
    }

    /**
     * Ajoute un enfant au début de l’élément.
     * Émet l’événement personnalisé `prepend`.
     */
    public prepend(child: any /*Block | HTMLElement*/) : Block {

        if (child instanceof Block)
            this.element.prepend(child.element);
        else if (child instanceof HTMLElement)
            this.element.prepend(child);

        this.emit('prepend');

        return this;
    }
    
    /**
     * Ajoute le bloc courant comme enfant d’un autre parent (`append`).
     * Émet `append` sur le parent si c’est un `Block`.
     */
    public appendTo(parent: any /*Block | HTMLElement*/) : Block { 

        if (parent instanceof Block)
            parent.element.append(this.element);
        else if (parent instanceof HTMLElement)
            parent.append(this.element);

        if (parent instanceof Block)
            parent.emit('append');

        return this;
    }
    
    /**
     * Ajoute le bloc courant au début d’un autre parent (`prepend`).
     * Émet `prepend` sur le parent si c’est un `Block`.
     */
    public prependTo(parent: any /*Block | HTMLElement*/) : Block {

        if (parent instanceof Block)
            parent.element.prepend(this.element);
        else if (parent instanceof HTMLElement)
            parent.prepend(this.element);
        
        if (parent instanceof Block)
            parent.emit('prepend');
        
        return this;
    }

    /** Vérifie si le bloc possède une classe CSS donnée. */
    public hasClass(className: string) : boolean {

        return this.element.classList.contains(className);
    }

    /**
     * Ajoute une ou plusieurs classes CSS à l’élément.
     * @example block.addClass('visible highlighted');
     */
    public addClass(className: string) : Block {
        
        if (className.trim() === '')
            return this;
        
        let classes = className.split(' ');

        for (let className_ of classes)
            this.element.classList.add(className_);

        return this;
    }

    /** Supprime une classe CSS de l’élément. */
    public removeClass(className: string) : Block {

        this.element.classList.remove(className);

        return this;
    }

    /**
     * Méthode appelée juste avant la suppression du bloc (à surcharger si besoin).
     * @example
     * ```typescript
     * class CustomBlock extends Block {
     *   onBeforeDelete() { console.log('Bloc supprimé'); }
     * }
     * ```
     */
    public onBeforeDelete(): void {}

    /**
     * Supprime l’élément du DOM et nettoie ses écouteurs d’événements.
     */
    public delete() : void {

        if (!this.element)
            return;
        
        if (typeof this.onBeforeDelete === 'function')
            this.onBeforeDelete();
        
        if (this.element.parentNode)
            this.element.parentNode.removeChild(this.element);
        
        this.clearListeners();
    }

    /** Vérifie si l’élément est actuellement monté dans le DOM. */
    public isMounted() : boolean {

        return document.contains(this.element);
    }

    /**
     * Ajoute un écouteur d’événement natif à l’élément HTML encapsulé.
     * @example block.onNative('click', () => console.log('Clicked'));
     */
    public onNative(key: string, callback: any) {

        this.element.addEventListener(key, callback/*, {
            passive: true
        }*/);

        return this;
    }

    /**
     * Crée et retourne un `Block` encapsulant un `DocumentFragment`.
     * Utile pour manipuler plusieurs éléments avant insertion dans le DOM.
     */
    static createFragment() : Block {

        return new Block(document.createDocumentFragment());
    }

    /**
     * Retourne l’index du bloc parmi ses frères dans le parent.
     */
    public getIndex() : number {

        return Array.prototype.indexOf.call(this.element.parentNode.children, this.element);
    }
}