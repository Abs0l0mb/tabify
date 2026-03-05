'use strict';

import {
    Block,
    Menu,
    Div,
    ClientLocation,
    AltBox
} from '@src/classes';

export interface MenuItemData {
    label: string
    path?: string
    requiredRight?: string
    extraClass?: string
    submenu?: SubmenuItemData[]
    submenuOpened?: boolean 
}

export interface SubmenuItemData {
    label: string,
    path: string,
    extraClass: string,
    requiredRight?: string
}

export class MenuItem extends Div {

    constructor(protected data: MenuItemData, protected menu: Menu, parent: Block) {

        super('menu-item', parent);

        if (this.data.requiredRight && !ClientLocation.get().api.accountData?.access_right_names?.includes(this.data.requiredRight)) {
            this.delete();
            return;
        }

        if (this.data.extraClass)
            this.addClass(this.data.extraClass);
        
        const content = new Div('content', this).onNative('click', this.route.bind(this));

        new AltBox(content, this.data.label, {
            className: 'navigation-alt-box'
        });

        new Div('text', content).write(this.data.label);

        ClientLocation.get().router.on('beforeRoute', (path: string) => {
            if (this.data.path && path.split('?')[0] === this.data.path)
                this.select();
            else
                this.unselect();
        });

        this.onNative('click', () => {
            this.menu.emit('item-click');
        });
    }

    /*
    **
    **
    */
    private select() : void {

        this.setData('selected', 1);
        this.emit
    }

    /*
    **
    **
    */
    public unselect() : void {

        this.setData('selected', 0);
    }

    /*
    **
    **
    */
    protected route() : void {

        console.log('route from MenuItem');
        
        ClientLocation.get().router.route(this.data.path);
    }
}